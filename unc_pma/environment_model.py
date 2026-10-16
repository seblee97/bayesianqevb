"""Explicit environment model (transitions + reward) used for planning.

Mattar & Daw (2018) "Prioritized memory access explains planning and
hippocampal replay" generate every planning backup from a learned model of
the environment rather than from a fixed replay buffer of past transitions.
This module makes that model a first-class, standalone object so that
PMAagent (unc_pma.agents.PMAagent) — or any future planning agent — can be
handed a different model implementation (e.g. one with uncertainty over
transitions/rewards) without changing the planner itself.
"""
from __future__ import annotations

from typing import Literal

import numpy as np


class TabularEnvironmentModel:
    """Learned model of a discrete-state, discrete-action MDP's dynamics.

    Maintains two independent pieces, both updated online from real
    experience via ``update()``:

      - ``T[s, a, :]``: a running estimate of the transition distribution
        P(s' | s, a), nudged toward each observed transition by
        ``t_learning_rate`` (1.0 -> always overwrite with the last
        transition; smaller values average over multiple visits). This is
        what the planner uses to compute the *Need* term (via the
        successor representation).

      - ``last_reward[s, a]`` / ``last_next_state[s, a]``: a one-sample
        cache of the most recently observed (r, s') for a given (s, a)
        pair. This is what the planner queries to synthesize hypothetical
        experiences for candidate backups. It deliberately remembers a
        single past sample rather than an average, so that planning
        replays an actual past transition rather than a blended,
        possibly-impossible one — matching the reference implementation.

    This class owns no Q-values and makes no action-selection decisions:
    it is a pure world model that a planning agent reads from and writes
    to, kept separate so it can later be swapped for a richer model (e.g.
    a Bayesian one with uncertainty over T and R) without touching the
    planning logic that consumes it.
    """

    def __init__(self, n_states: int, n_actions: int, t_learning_rate: float = 0.9):
        self.n_states = n_states
        self.n_actions = n_actions
        self.t_learning_rate = t_learning_rate

        self.T = np.zeros((n_states, n_actions, n_states))
        self.last_reward = np.full((n_states, n_actions), np.nan)
        self.last_next_state = np.full((n_states, n_actions), -1, dtype=int)
        self.visited = np.zeros((n_states, n_actions), dtype=bool)

    # ------------------------------------------------------------------
    # Learning the model from real experience
    # ------------------------------------------------------------------

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        """Incorporate one observed (state, action, reward, next_state)."""
        target = np.zeros(self.n_states)
        target[next_state] = 1.0
        self.T[state, action] += self.t_learning_rate * (target - self.T[state, action])
        self.last_reward[state, action] = reward
        self.last_next_state[state, action] = next_state
        self.visited[state, action] = True

    # ------------------------------------------------------------------
    # Querying the model for planning
    # ------------------------------------------------------------------

    def predict(self, state: int, action: int) -> tuple[float, int] | None:
        """Return the model's remembered (reward, next_state) for (s, a).

        Returns None if (state, action) has never been observed.
        """
        if not self.visited[state, action]:
            return None
        return float(self.last_reward[state, action]), int(self.last_next_state[state, action])

    def known_state_actions(self) -> list[tuple[int, int]]:
        """All (state, action) pairs the model has observed at least once."""
        ss, aa = np.where(self.visited)
        return list(zip(ss.tolist(), aa.tolist()))

    def set_transition(self, state: int, action: int, next_state_probs: np.ndarray) -> None:
        """Directly set T[state, action, :] to a given distribution over next states.

        Unlike `update()`, this bypasses the exponential-smoothing rule and
        does *not* mark (state, action) as visited or touch the one-step
        reward/next_state cache used for candidate generation. It exists
        for seeding transitions that are not real chosen actions -- e.g.
        an episodic environment's goal-to-next-start transition, which
        the reference implementation includes in the transition model
        (so the Need term correctly anticipates the agent's return) while
        explicitly excluding it from ever being offered as a planning
        candidate (there is no real action "at" the goal to replay).
        """
        self.T[state, action] = next_state_probs

    def successor_representation(self, gamma: float, policy_probs: np.ndarray) -> np.ndarray:
        """SR(s, s') = expected discounted visits to s' starting from s.

        SR = (I - gamma * T_pi)^-1, where the on-policy state-to-state
        transition matrix T_pi(s, s') = sum_a pi(a|s) T(s, a, s') is built
        by marginalizing the learned per-action model over a policy.

        Args:
            gamma: discount factor.
            policy_probs: (n_states, n_actions) array of action
                probabilities under the policy whose induced Markov chain
                the successor representation describes.
        """
        t_pi = np.einsum("sa,sap->sp", policy_probs, self.T)
        return np.linalg.inv(np.eye(self.n_states) - gamma * t_pi)


class DirichletTransitionsModel:
    """Dirichlet-Categorical posterior over P(s' | s, a), for Thompson-sampling planners.

    Bai, Wu & Chen (2013) "Bayesian Mixture Modelling and Inference based
    Thompson Sampling in Monte-Carlo Tree Search" (NIPS 2013; see
    papers/NIPS-2013-bayesian-mixture-modelling-...-Paper-2.pdf) place a
    Dirichlet prior Dir(rho_{s,a}) over the unknown next-state weights
    w_{s,a,s'} = T(s' | s, a), since the Dirichlet is the conjugate prior
    of a discrete distribution. Each observed transition (s, a) -> s' is
    a one-hot observation from that discrete distribution, so the
    posterior update is simply rho_{s,a,s'} <- rho_{s,a,s'} + 1 (Eq. in
    Sec. 3.2 of the paper); rho_{s,a,s'} - 1 is thus the number of times
    s' has followed (s, a). The paper initializes rho_{s,a,s'} to a small
    positive constant delta for an (approximately) uninformative prior.

    Where TabularEnvironmentModel keeps a single point estimate of T,
    this class keeps the full Dirichlet concentration parameters, so a
    planner can either read the posterior mean or draw a posterior sample
    (as in the paper's Thompson-sampling branch of QValue/ThompsonSampling,
    which samples w_{s,a,:} ~ Dir(rho_{s,a}) rather than using its mean).
    """

    def __init__(self, n_states: int, n_actions: int, alpha_prior: float = 1e-3):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha_prior = alpha_prior

        self.alpha = np.full((n_states, n_actions, n_states), alpha_prior)
        self.visited = np.zeros((n_states, n_actions), dtype=bool)

    # ------------------------------------------------------------------
    # Learning the model from real experience
    # ------------------------------------------------------------------

    def update(self, state: int, action: int, next_state: int) -> None:
        """Record one observed (state, action) -> next_state transition.

        Adds one count to the Dirichlet concentration parameter of the
        observed next state, rho_{s,a,s'} <- rho_{s,a,s'} + 1, leaving the
        concentration for every other next state unchanged.
        """
        self.alpha[state, action, next_state] += 1.0
        self.visited[state, action] = True

    def mark_dead_end(self, states) -> None:
        """Force every `state` in `states` to self-loop under every action.

        For states the agent can never actually occupy (e.g. wall cells in
        a gridworld), the default flat `alpha_prior` is worse than merely
        unused: a planner that looks *backward* from a real state (e.g.
        `expected_prev_state_evb` in `ReplaySamplerAgent`) scans every
        `(state, action)` as a candidate predecessor, including these dead
        states. Left at the flat prior, they'd report a small but nonzero
        posterior-mean probability of transitioning into real states --
        which, multiplied against the large-magnitude sentinel a planner
        typically uses to mask a dead state's own value (e.g. `-1e6` for
        `valid_states`-excluded states), injects a spurious, wildly
        out-of-scale contribution into that computation. Pinning `P(state |
        state, a) = 1` for every `a` (and 0 for every other destination)
        makes a dead state contribute exactly 0 to any real state's
        backward-predecessor sum instead.

        Call this *after* any other seeding of these states' rows, since
        it overwrites them outright; it does not need to run after seeding
        of *other* (non-dead) states' rows, since it only ever touches the
        rows belonging to `states`.
        """
        states = np.asarray(list(states), dtype=int)
        self.alpha[states, :, :] = 0.0
        self.alpha[states, :, states] = 1.0

    # ------------------------------------------------------------------
    # Querying the model
    # ------------------------------------------------------------------

    def mean_transition_probs(self, state: int, action: int) -> np.ndarray:
        """Posterior mean E[w_{s,a,:}] = rho_{s,a,:} / sum(rho_{s,a,:})."""
        concentration = self.alpha[state, action]
        return concentration / concentration.sum()

    def sample_next_state(self, state: int, action: int) -> int:
        """Thompson-sample a next state for (state, action).

        Draws one posterior sample of the transition distribution,
        w ~ Dir(rho_{s,a,:}), then draws next_state ~ Categorical(w) --
        matching the paper's posterior-sampling branch (`sampling=True`)
        of QValue/ThompsonSampling, rather than acting on the posterior
        mean.
        """
        probs = np.random.dirichlet(self.alpha[state, action])
        return int(np.random.choice(self.n_states, p=probs))

    def successor_representation(self, gamma: float, policy_probs: np.ndarray) -> np.ndarray:
        """SR(s, s') = expected discounted visits to s' starting from s.

        Same construction as TabularEnvironmentModel.successor_representation:
        SR = (I - gamma * T_pi)^-1, where T_pi(s, s') = sum_a pi(a|s) T(s, a, s')
        marginalizes a per-action transition estimate over the policy. Here
        T(s, a, :) is taken to be the Dirichlet posterior mean
        (`mean_transition_probs`), since the successor representation is a
        point-estimate (matrix-inverse) quantity rather than something a
        single posterior sample should be plugged into.

        Args:
            gamma: discount factor.
            policy_probs: (n_states, n_actions) array of action
                probabilities under the policy whose induced Markov chain
                the successor representation describes.
        """
        mean_t = self.alpha / self.alpha.sum(axis=-1, keepdims=True)
        t_pi = np.einsum("sa,sap->sp", policy_probs, mean_t)
        return np.linalg.inv(np.eye(self.n_states) - gamma * t_pi)


class RewardModel:
    """Posterior over the expected immediate reward E[r | s, a], one per (s, a).

    Replay builds n-step targets from rewards the agent *believes* it would
    collect, so the replay sampler needs beliefs over R alongside the
    Dirichlet beliefs over T. Two observation models are offered:

      - noise="unknown" (default): Normal-Gamma over (mean, precision) of
        the reward, the same conjugate family as the Q posteriors
        (unc_pma.bayesian_qlearning.NormalGamma), so the reward noise level
        is learned rather than supplied.
      - noise="known": Normal over the mean with a fixed observation noise
        variance `noise_var`, e.g. the environment's reward_noise ** 2.

    Either way, what replay consumes is the posterior over the *mean*
    reward: `mean()` and `mean_variance()`, the epistemic variance of
    E[r | s, a] (not the spread of individual rewards).

    Args:
        n_states, n_actions: size of the discrete state/action space.
        noise: "unknown" or "known", see above.
        noise_var: observation noise variance, only used if noise="known".
        mu0, lam0: prior mean and pseudo-count shared by both models.
        alpha0, beta0: Normal-Gamma Gamma shape/rate, only used if
            noise="unknown". The default alpha0=2 keeps the marginal
            variance finite before any observation.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        noise: Literal["unknown", "known"] = "unknown",
        noise_var: float = 1.0,
        mu0: float = 0.0,
        lam0: float = 1.0,
        alpha0: float = 2.0,
        beta0: float = 1.0,
    ):
        if noise not in ("unknown", "known"):
            raise ValueError(f"Unknown noise model {noise!r}")
        self.n_states = n_states
        self.n_actions = n_actions
        self.noise = noise
        self.noise_var = noise_var

        self.mu = np.full((n_states, n_actions), float(mu0))
        self.lam = np.full((n_states, n_actions), float(lam0))
        self.alpha = np.full((n_states, n_actions), float(alpha0))
        self.beta = np.full((n_states, n_actions), float(beta0))

    def update(self, state: int, action: int, reward: float) -> None:
        """Conjugate update of the (state, action) posterior with one observed reward.

        Same update as NormalGamma.update; the known-noise model only moves
        mu and lam.
        """
        mu, lam = self.mu[state, action], self.lam[state, action]
        lam_new = lam + 1.0
        if self.noise == "unknown":
            self.alpha[state, action] += 0.5
            self.beta[state, action] += lam * (reward - mu) ** 2 / (2.0 * lam_new)
        self.mu[state, action] = (lam * mu + reward) / lam_new
        self.lam[state, action] = lam_new

    def mean(self) -> np.ndarray:
        """(n_states, n_actions) posterior mean of E[r | s, a]."""
        return self.mu

    def mean_variance(self) -> np.ndarray:
        """(n_states, n_actions) posterior variance of E[r | s, a].

        Normal-Gamma: variance of the Student-t marginal of the mean,
        beta / ((alpha - 1) * lam). It is infinite for alpha <= 1, where the
        squared t-scale beta / (alpha * lam) is used instead so it stays
        usable in moment propagation.
        """
        if self.noise == "known":
            return self.noise_var / self.lam
        return np.where(
            self.alpha > 1.0,
            self.beta / (np.maximum(self.alpha - 1.0, 1e-12) * self.lam),
            self.beta / (self.alpha * self.lam),
        )


class TerminalModel:
    """Belief over which states end the episode when entered.

    Replay rolls out the agent's model, not the environment, so no `done`
    flag arrives during a rollout; whether entering s' ends the episode has
    to come from beliefs. It is consumed as a transition-dependent discount
    gamma(s') = gamma * P(continue | s'), which is 0 at a terminal state
    (White 2017, "Unifying Task Specification in Reinforcement Learning").
    Combined with the transition posterior, the effective discount of
    (s, a) is gamma * sum_s' T(s'|s, a) P(continue | s'), so uncertainty
    about *reaching* a terminal comes from T for free.

    Two modes:
      - mode="known" (default): `terminal_states` (boolean mask or indices)
        is taken as known. P(continue) is 0 there and 1 elsewhere.
      - mode="learned": a Beta(a, b) belief per state over P(terminal),
        updated from the `done` flags of real transitions via `update()`.
        The default prior (a0=1e-3, b0=1) starts every state as almost
        certainly non-terminal.
    """

    def __init__(
        self,
        n_states: int,
        mode: Literal["known", "learned"] = "known",
        terminal_states=None,
        a0: float = 1e-3,
        b0: float = 1.0,
    ):
        if mode not in ("known", "learned"):
            raise ValueError(f"Unknown terminal mode {mode!r}")
        self.n_states = n_states
        self.mode = mode

        self.terminal = np.zeros(n_states, dtype=bool)
        if terminal_states is not None:
            terminal_states = np.asarray(terminal_states)
            if terminal_states.dtype == bool:
                self.terminal = terminal_states.copy()
            else:
                self.terminal[terminal_states.astype(int)] = True

        self.a = np.full(n_states, float(a0))
        self.b = np.full(n_states, float(b0))

    def update(self, next_state: int, done: bool) -> None:
        """Record whether entering `next_state` ended the episode (learned mode only)."""
        if self.mode != "learned":
            return
        if done:
            self.a[next_state] += 1.0
        else:
            self.b[next_state] += 1.0

    def continue_mean(self) -> np.ndarray:
        """(n_states,) E[P(continue | s')]."""
        if self.mode == "known":
            return (~self.terminal).astype(float)
        return self.b / (self.a + self.b)

    def continue_var(self) -> np.ndarray:
        """(n_states,) Var[P(continue | s')] -- zero in known mode."""
        if self.mode == "known":
            return np.zeros(self.n_states)
        total = self.a + self.b
        return self.a * self.b / (total ** 2 * (total + 1.0))
