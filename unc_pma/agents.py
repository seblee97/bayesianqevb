"""Prioritized Memory Access (PMA) agent.

Reference: Mattar, M. G., & Daw, N. D. (2018). "Prioritized memory access
explains planning and hippocampal replay." Nature Neuroscience.
Companion code: https://github.com/marcelomattar/PrioritizedReplay

PMA interleaves ordinary Q-learning on real experience with *planning*:
after every real step, the agent replays hypothetical experiences drawn
from a learned model of the environment (unc_pma.environment_model.
TabularEnvironmentModel), prioritized by their Expected Value of Backup
(EVB = Gain x Need):

  - Gain(s, a):  how much the greedy policy at s improves if Q(s, a) is
                 updated toward a candidate target (Dyna-style value of
                 a backup).
  - Need(s):     how often s is expected to be visited from the agent's
                 current state, i.e. the successor representation induced
                 by the current policy under the learned transition model.

The highest-EVB candidate is backed up for real, and the planner then
tries to *chain* one more transition onto the end of it (sampling the
model at the state that backup arrived at), so that a single planning
bout can string together a multi-step trajectory rather than only ever
replaying isolated one-step transitions.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from unc_pma.environment_model import TabularEnvironmentModel
from unc_pma.replay_buffer import Experience

if TYPE_CHECKING:
    from unc_pma.environments import Environment


class PMAagent:
    """Prioritized Memory Access agent (Mattar & Daw, 2018).

    Args:
        n_states, n_actions: size of the discrete state/action space.
        gamma:      discount factor, shared by real learning and planning.
        alpha:      Q-learning step size, shared by real learning and planning.
        model:      a TabularEnvironmentModel instance, or None to create
                    a fresh one. Passing an explicit model is what makes
                    the model swappable (e.g. for a future model with
                    uncertainty over T and R) without changing PMAagent.
        t_learning_rate: learning rate for the model's transition
                    estimate, only used if `model` is None.
        policy:     'softmax' or 'epsilon_greedy' — used both for real
                    action selection and, if `off_policy=False`, for
                    bootstrapping values and sampling chained transitions.
        softmax_beta, epsilon: policy parameters.
        off_policy: if True, bootstrap with max_a Q(s',a) (Q-learning,
                    matches the reference default); if False, bootstrap
                    with the expected value under `policy` (Expected
                    SARSA-style, on-policy planning/learning).
        n_plan:     max number of planning backups to attempt per planning bout.
        auto_plan:  if True (default), `update()` triggers a planning bout
                    of up to `n_plan` backups after every real step (Dyna-
                    style). If False, `update()` only performs the real
                    Q-learning update and model update; the caller is
                    responsible for invoking `.plan()` explicitly (e.g. to
                    match Mattar & Daw's setup of planning only at the
                    start and end of each episode).
        evb_threshold: minimum EVB required to perform a planning backup;
                    planning for this step stops once no candidate clears it.
        baseline_gain: gain is floored at this value before weighting by
                    need, so that a purely-informative (zero-gain) backup
                    can still register a small nonzero EVB.
        expand_further: if True, try to extend the previous winning
                    trajectory by one more transition each planning
                    iteration (the "chaining" mechanism), competing it
                    against fresh one-step candidates.
        allow_loops: if False, a chained extension that revisits a state
                    already in the trajectory is discarded.
        skip_self_transitions: if True, (s, a) pairs whose modeled
                    next_state == s (e.g. bumping into a wall) are not
                    offered as one-step planning candidates.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = 0.9,
        alpha: float = 1.0,
        model: TabularEnvironmentModel | None = None,
        t_learning_rate: float = 0.9,
        policy: str = "softmax",
        softmax_beta: float = 5.0,
        epsilon: float = 0.05,
        off_policy: bool = True,
        n_plan: int = 20,
        auto_plan: bool = True,
        evb_threshold: float = 0.0,
        baseline_gain: float = 1e-10,
        expand_further: bool = True,
        allow_loops: bool = False,
        skip_self_transitions: bool = True,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha

        self.model = model if model is not None else TabularEnvironmentModel(
            n_states, n_actions, t_learning_rate
        )

        self.policy = policy
        self.softmax_beta = softmax_beta
        self.epsilon = epsilon
        self.off_policy = off_policy

        self.n_plan = n_plan
        self.auto_plan = auto_plan
        self.evb_threshold = evb_threshold
        self.baseline_gain = baseline_gain
        self.expand_further = expand_further
        self.allow_loops = allow_loops
        self.skip_self_transitions = skip_self_transitions

        self.Q = np.zeros((n_states, n_actions))
        self._chain: list[Experience] = []
        self._current_state = 0
        self.last_plan: list[tuple[list[Experience], float, float]] = []
        self.last_sr: np.ndarray | None = None
        self.last_plan_origin_state: int = 0

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def _policy_probs(self, q_row: np.ndarray) -> np.ndarray:
        """Action-selection probabilities for one state's Q-values."""
        if self.policy == "softmax":
            z = self.softmax_beta * (q_row - q_row.max())
            exp_z = np.exp(z)
            return exp_z / exp_z.sum()
        elif self.policy == "epsilon_greedy":
            probs = np.full(self.n_actions, self.epsilon / self.n_actions)
            best = np.flatnonzero(q_row == q_row.max())
            probs[best] += (1.0 - self.epsilon) / len(best)
            return probs
        raise ValueError(f"Unknown policy {self.policy!r}")

    def _bootstrap_value(self, state: int) -> float:
        """Value used to bootstrap n-step returns: max (off-policy) or expected (on-policy)."""
        q_row = self.Q[state]
        if self.off_policy:
            return float(np.max(q_row))
        probs = self._policy_probs(q_row)
        return float(np.dot(probs, q_row))

    def select_action(self, state: int) -> int:
        probs = self._policy_probs(self.Q[state])
        return int(np.random.choice(self.n_actions, p=probs))

    def greedy_action(self, state: int) -> int:
        return int(np.argmax(self.Q[state]))

    def value(self, state: int) -> float:
        return float(np.max(self.Q[state]))

    # ------------------------------------------------------------------
    # Real experience
    # ------------------------------------------------------------------

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        """Learn from one real transition, update the model, then (optionally) plan.

        Mirrors the reference loop: real Q-learning update first (so the
        model and Q-table reflect this transition), then, if `auto_plan`
        is set, a bout of planning backups drawn from the model, anchored
        at `next_state` (the state the agent now occupies). With
        `auto_plan=False`, planning is left entirely to explicit calls to
        `.plan()` (e.g. only at episode boundaries).
        """
        self.model.update(state, action, reward, next_state)

        target = reward if done else reward + self.gamma * self._bootstrap_value(next_state)
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

        self._current_state = next_state
        if self.auto_plan:
            self._chain = []
            self.plan()

    # ------------------------------------------------------------------
    # Gain and Need
    # ------------------------------------------------------------------

    def _gain(self, state: int, action: int, q_target: float) -> float:
        """Change in expected value at `state` from backing up Q(state, action) to q_target.

        Uses the OLD policy weighted against the NEW (post-backup) values
        for the "pre" term, and the NEW policy against the NEW values for
        the "post" term — Gain = E_pi_new[Q_new] - E_pi_old[Q_new].
        """
        q_row = self.Q[state].copy()
        probs_pre = self._policy_probs(q_row)
        q_row[action] = q_target
        probs_post = self._policy_probs(q_row)
        ev_pre = float(np.dot(probs_pre, q_row))
        ev_post = float(np.dot(probs_post, q_row))
        return ev_post - ev_pre

    def _n_step_targets(self, trajectory: list[Experience]) -> list[tuple[int, int, float]]:
        """Q-learning target for every prefix of `trajectory`, bootstrapped off its shared end.

        All prefixes j..end share the same bootstrap value (the value of
        the trajectory's final next_state); only the discounted reward
        run and the discount power differ, matching an n-step return
        computed once per trajectory rather than once per prefix.
        """
        stp1_value = self._bootstrap_value(trajectory[-1].next_state)
        n = len(trajectory)
        targets = []
        for j in range(n):
            discounted_return = sum(
                self.gamma ** k * exp.reward for k, exp in enumerate(trajectory[j:])
            )
            steps_remaining = n - j
            q_target = discounted_return + self.gamma ** steps_remaining * stp1_value
            targets.append((trajectory[j].state, trajectory[j].action, q_target))
        return targets

    def _step_gains(self, trajectory: list[Experience]) -> list[float]:
        """Raw (unfloored) Gain for every prefix of `trajectory`, in order."""
        return [self._gain(s, a, q_target) for s, a, q_target in self._n_step_targets(trajectory)]

    def _trajectory_evb(self, trajectory: list[Experience], sr: np.ndarray) -> float:
        """EVB of backing up every prefix of `trajectory` toward its shared n-step return.

        Need is taken from the *origin* state of the trajectory's last
        (most recently appended) transition, so a chained multi-step
        trajectory is weighted by how likely the agent is to reach the
        point it has grown to, while Gain accumulates over every backup
        the trajectory performs.
        """
        total_gain = sum(max(g, self.baseline_gain) for g in self._step_gains(trajectory))
        origin_state = trajectory[-1].state
        need = float(sr[self._current_state, origin_state])
        return need * total_gain

    def _apply_backup(self, trajectory: list[Experience]) -> None:
        """Actually perform the Q-learning update for every prefix of `trajectory`."""
        for s, a, q_target in self._n_step_targets(trajectory):
            self.Q[s, a] += self.alpha * (q_target - self.Q[s, a])

    # ------------------------------------------------------------------
    # Candidate generation, including the chaining mechanism
    # ------------------------------------------------------------------

    def _sample_model_action(self, state: int) -> int:
        """Action used to extend a chain from `state`, following the bootstrap policy."""
        q_row = self.Q[state]
        if self.off_policy:
            best = np.flatnonzero(q_row == q_row.max())
            return int(np.random.choice(best))
        probs = self._policy_probs(q_row)
        return int(np.random.choice(self.n_actions, p=probs))

    def _build_candidates(self) -> list[list[Experience]]:
        """One-step candidates from every known (s, a), plus one chained extension."""
        candidates: list[list[Experience]] = []
        for s, a in self.model.known_state_actions():
            prediction = self.model.predict(s, a)
            if prediction is None:
                continue
            r, s_next = prediction
            if self.skip_self_transitions and s_next == s:
                continue
            candidates.append([Experience(s, a, r, s_next, False)])

        if self.expand_further and self._chain:
            last_state = self._chain[-1].next_state
            a_n = self._sample_model_action(last_state)
            prediction = self.model.predict(last_state, a_n)
            if prediction is not None:
                r_n, s_next = prediction
                is_self_transition = self.skip_self_transitions and s_next == last_state
                visited_states = {exp.state for exp in self._chain} | {last_state}
                if not is_self_transition and (self.allow_loops or s_next not in visited_states):
                    candidates.append(self._chain + [Experience(last_state, a_n, r_n, s_next, False)])

        return candidates

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan(self, n_steps: int | None = None) -> list[tuple[list[Experience], float, float]]:
        """Run a bout of prioritized planning backups, chaining trajectories where useful.

        Each iteration: build candidate (possibly multi-step) backups from
        the model, score each by EVB = Need x Gain, execute the highest-
        scoring one if it clears `evb_threshold`, then try to extend that
        same trajectory by one more transition on the next iteration.
        Stops early once no candidate's EVB clears the threshold.

        Returns a list of (trajectory, evb, gain) per executed backup, where
        `gain` is the raw Gain of just the newly-added link (trajectory[-1]) —
        useful for e.g. visualizing which state each backup most improved.
        """
        n_steps = self.n_plan if n_steps is None else n_steps
        self._chain = []
        executed: list[tuple[list[Experience], float, float]] = []

        if not self.model.visited.any():
            self.last_plan = executed
            return executed

        policy_probs_table = np.array(
            [self._policy_probs(self.Q[s]) for s in range(self.n_states)]
        )
        sr = self.model.successor_representation(self.gamma, policy_probs_table)
        self.last_sr = sr
        self.last_plan_origin_state = self._current_state

        for _ in range(n_steps):
            candidates = self._build_candidates()
            if not candidates:
                break

            evbs = np.array([self._trajectory_evb(traj, sr) for traj in candidates])
            best_evb = float(evbs.max())
            if best_evb <= self.evb_threshold:
                break

            tied = np.flatnonzero(evbs == best_evb)
            if len(tied) > 1:
                lengths = np.array([len(candidates[i]) for i in tied])
                tied = tied[lengths == lengths.min()]
            best_idx = int(np.random.choice(tied)) if len(tied) > 1 else int(tied[0])

            best_trajectory = candidates[best_idx]
            new_link_gain = self._step_gains(best_trajectory)[-1]
            self._apply_backup(best_trajectory)
            self._chain = best_trajectory
            executed.append((best_trajectory, best_evb, new_link_gain))

        self.last_plan = executed
        return executed

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def pre_explore(self, env: "Environment", n_steps: int) -> None:
        """Randomly explore `env` to seed the transition/reward model, without Q-updates.

        Mirrors the reference implementation's `preExplore` step: the
        model needs to know the consequences of actions before planning
        (Gain/Need) can be computed meaningfully.
        """
        state = env.reset()
        for _ in range(n_steps):
            action = int(np.random.randint(self.n_actions))
            next_state, reward, done, _ = env.step(action)
            self.model.update(state, action, reward, next_state)
            state = next_state if not done else env.reset()

    def run_episode(self, env: "Environment", max_steps: int = 1000) -> float:
        """Run one episode, return total undiscounted reward."""
        state = env.reset()
        total_reward = 0.0
        for _ in range(max_steps):
            action = self.select_action(state)
            next_state, reward, done, _ = env.step(action)
            self.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                break
        return total_reward

