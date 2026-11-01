"""Belief-guided rollout proposer for ReplaySamplerAgent.

A simpler alternative to the EVB-guided chain search: candidate replays are
posterior-sampled rollouts, in the spirit of Thompson sampling in tree search
(Bai, Wu & Chen 2013), and EVB is used only to choose among them.

Each rollout draws one world from the beliefs -- T~ ~ Dirichlet for every
(s, a) and Q~ ~ Normal-Gamma for every (s, a) -- and keeps it for the whole
rollout:

  * root: s_0 from `root` (Need by default), a_0 = argmax_a Q~(s_0, a);
  * forward (with probability `forward_prob`): s' ~ T~(.|s, a), stopping
    with probability 1 - P(continue | s') from the TerminalModel, then
    a' = argmax_a Q~(s', a);
  * backward: a predecessor (s', a') of the earliest node's state s, drawn
    with weight T~(s | s', a') pi~(a'|s'), pi~ greedy in Q~ -- falling back
    to T~ alone when no predecessor's sampled-greedy action leads to s.

A rollout never revisits a state, follows the same legality rules as the
EVB chain (valid pairs only; going forward, no action whose dominant
successor is already in the rollout) and stops when no legal move is left or
after `max_length` nodes.

Every prefix of a rollout (lengths 1, ..., L) is offered as its own
candidate, as Mattar & Daw do for n-step backups. On its own that rarely
shortens a replay: total EVB never falls as nodes are added for non-negative
gains, so a prefix only wins when the rest of the rollout adds exactly
nothing (never for gain="vpi", whose VPI is positive everywhere). Set the
agent's `backup_cost` so that each node must earn its place.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from unc_pma.replay_buffer import StateAction
from unc_pma.replay_sampling_agent import Candidate, _ng_variance

if TYPE_CHECKING:
    from unc_pma.replay_sampling_agent import ReplaySamplerAgent, _Bout

RootDistribution = Literal["need", "agent_state", "uncertainty", "uniform"]


class RolloutProposer:
    """Posterior-sampled rollouts, every prefix a candidate. See the module docstring.

    Args:
        root: where rollouts start --
            "need": s_0 ~ Need(s) (the bout's Need, i.e. the SR row of the
                agent's real state, or the stationary distribution);
            "agent_state": always the agent's real state;
            "uncertainty": s_0 proportional to sum_a Var[Q(s, a)] over the
                state's valid actions;
            "uniform": uniform over states with a valid action.
            Restricted to states with at least one valid action; if that
            leaves no mass (e.g. Need concentrated on terminal states),
            falls back to uniform.
        forward_prob: probability that a rollout goes forward rather than
            backward from its root.
        max_length: cap on rollout length in nodes; None means no cap
            beyond the no-revisit rule (at most n_states nodes).
    """

    def __init__(self, root: RootDistribution = "need", forward_prob: float = 0.5,
                 max_length: int | None = None):
        if root not in ("need", "agent_state", "uncertainty", "uniform"):
            raise ValueError(f"Unknown root {root!r}")
        if not 0.0 <= forward_prob <= 1.0:
            raise ValueError(f"forward_prob must be in [0, 1], got {forward_prob!r}")
        if max_length is not None and max_length < 1:
            raise ValueError(f"max_length must be at least 1, got {max_length!r}")
        self.root = root
        self.forward_prob = forward_prob
        self.max_length = max_length

    def propose(self, agent: ReplaySamplerAgent, bout: _Bout, temp: float, n: int,
                return_logs: bool) -> list[Candidate]:
        """Every prefix of `n` independent rollouts. `temp` is unused: nothing here is a softmax over EVB."""
        q = agent.q_model
        ngs = [[q.q_params(s, a) for a in range(q.n_actions)] for s in range(q.n_states)]
        q_post = {key: np.array([[getattr(ng, key) for ng in row] for row in ngs])
                  for key in ("mu", "lam", "alpha", "beta")}
        root_probs = self._root_probs(agent, bout, q_post)

        candidates = []
        for _ in range(n):
            tree, strategy = self._rollout(agent, bout, q_post, root_probs)
            for k in range(1, len(tree) + 1):
                logs = {"rollout_length": len(tree)} if return_logs else None
                candidates.append((tree[:k], strategy[:k], logs))
        return candidates

    def _root_probs(self, agent: ReplaySamplerAgent, bout: _Bout, q_post: dict) -> np.ndarray:
        """(n_states,) distribution s_0 is drawn from."""
        valid_roots = bout.valid_pair_mask.any(axis=1)
        if self.root == "agent_state":
            s = int(agent.agent_real_state)
            if not valid_roots[s]:
                raise ValueError(f"agent_real_state {s} has no valid action to root a rollout at")
            probs = np.zeros(len(valid_roots))
            probs[s] = 1.0
            return probs

        if self.root == "need":
            weights = bout.need * valid_roots
        elif self.root == "uncertainty":
            q_var = _ng_variance(q_post["lam"], q_post["alpha"], q_post["beta"])
            weights = (q_var * bout.valid_pair_mask).sum(axis=1)
        else:  # "uniform"
            weights = valid_roots.astype(float)
        if not weights.sum() > 0.0:
            weights = valid_roots.astype(float)
        return weights / weights.sum()

    def _rollout(self, agent: ReplaySamplerAgent, bout: _Bout, q_post: dict,
                 root_probs: np.ndarray) -> tuple[list[StateAction], list[str]]:
        """One rollout in one posterior-sampled world, as (tree, strategy) in append order."""
        n_s = agent.q_model.n_states
        valid = bout.valid_pair_mask
        t_sample = _sample_transitions(agent.transition_model.alpha)
        q_sample = q_post["mu"] + np.sqrt(q_post["beta"] / (q_post["alpha"] * q_post["lam"])) \
            * np.random.standard_t(2.0 * q_post["alpha"])
        q_sample = np.where(valid, q_sample, -np.inf)
        # As in the EVB chain: the transition model is near-deterministic,
        # so its argmax stands in for "the state this (s, a) leads to".
        dominant_next_state = agent.transition_model.alpha.argmax(axis=-1)
        cont = bout.cont_mean if bout.cont_mean is not None else (
            agent.terminal_model.continue_mean() if agent.terminal_model is not None else np.ones(n_s))
        max_length = n_s if self.max_length is None else self.max_length

        s = int(np.random.choice(n_s, p=root_probs))
        a = int(np.argmax(q_sample[s]))
        tree, strategy = [StateAction(s, a)], ["root"]
        direction = "forward" if np.random.random() < self.forward_prob else "backward"

        while len(tree) < max_length:
            visited = [sa.state for sa in tree]
            if direction == "forward":
                s_next = int(np.random.choice(n_s, p=t_sample[s, a]))
                if np.random.random() >= cont[s_next] or s_next in visited:
                    break
                allowed = valid[s_next] & ~np.isin(dominant_next_state[s_next], visited)
                if not allowed.any():
                    break
                s, a = s_next, int(np.argmax(np.where(allowed, q_sample[s_next], -np.inf)))
            else:
                # tree[-1] is the earliest node in time; look for what leads into it.
                weights = t_sample[:, :, tree[-1].state] * valid
                weights[visited, :] = 0.0
                greedy = np.zeros_like(weights)
                greedy[np.arange(n_s), q_sample.argmax(axis=1)] = 1.0
                on_policy = weights * greedy
                if on_policy.sum() > 0.0:
                    weights = on_policy
                total = weights.sum()
                if total <= 0.0:
                    break
                index = np.random.choice(weights.size, p=(weights / total).ravel())
                s, a = agent.state_action_from_index(index)
            tree.append(StateAction(s, a))
            strategy.append(direction)
        return tree, strategy


def _sample_transitions(alpha: np.ndarray) -> np.ndarray:
    """One draw T~ ~ Dirichlet(alpha[s, a]) for every (s, a), via normalized Gamma draws.

    Unlike np.random.dirichlet this allows zero concentrations (an exactly
    impossible successor stays impossible). A row whose draws all underflow
    to zero -- possible with a tiny flat prior -- falls back to its
    posterior mean.
    """
    draws = np.random.gamma(alpha)
    totals = draws.sum(axis=-1, keepdims=True)
    mean = alpha / alpha.sum(axis=-1, keepdims=True)
    return np.where(totals > 0.0, draws / np.where(totals > 0.0, totals, 1.0), mean)
