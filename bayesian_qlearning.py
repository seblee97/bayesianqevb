"""Bayesian Q-learning with Normal-Gamma priors and VPI-based exploration.

Reference: Dearden, Friedman & Russell (1998) "Bayesian Q-learning", AAAI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from environments import Environment


# ---------------------------------------------------------------------------
# Normal-Gamma posterior
# ---------------------------------------------------------------------------

@dataclass
class NormalGamma:
    """Conjugate prior for a Gaussian with unknown mean and precision.

    The joint prior is:
        precision τ ~ Gamma(alpha, beta)        [rate parameterisation]
        mean      μ | τ ~ N(mu, 1 / (lam * τ))

    The marginal distribution of the mean is Student-t:
        Q ~ t(df=2α, loc=μ, scale=sqrt(β/(αλ)))

    Bayesian update for a new scalar observation x uses the standard
    Normal-Gamma conjugate update:
        λ' = λ + 1
        μ' = (λμ + x) / λ'
        α' = α + ½
        β' = β + λ(x − μ)² / (2λ')
    """

    mu: float = 0.0     # prior / posterior mean
    lam: float = 1.0    # precision multiplier (acts as a pseudo-count)
    alpha: float = 1.0  # Gamma shape
    beta: float = 1.0   # Gamma rate

    # ------------------------------------------------------------------
    # Posterior update
    # ------------------------------------------------------------------

    def update(self, x: float) -> NormalGamma:
        """Return a new NormalGamma updated with observation x."""
        lam_new = self.lam + 1.0
        mu_new = (self.lam * self.mu + x) / lam_new
        alpha_new = self.alpha + 0.5
        beta_new = self.beta + self.lam * (x - self.mu) ** 2 / (2.0 * lam_new)
        return NormalGamma(mu=mu_new, lam=lam_new, alpha=alpha_new, beta=beta_new)

    # ------------------------------------------------------------------
    # Marginal t-distribution properties
    # ------------------------------------------------------------------

    @property
    def df(self) -> float:
        return 2.0 * self.alpha

    @property
    def scale(self) -> float:
        """Scale parameter of the marginal t-distribution."""
        return float(np.sqrt(self.beta / (self.alpha * self.lam)))

    @property
    def mean(self) -> float:
        return self.mu

    @property
    def variance(self) -> float:
        """Variance of the marginal t (finite only for alpha > 1)."""
        if self.alpha <= 1.0:
            return float("inf")
        return float(self.beta / ((self.alpha - 1.0) * self.lam))

    def sample(self) -> float:
        return float(self.mu + self.scale * stats.t.rvs(df=self.df))

    # ------------------------------------------------------------------
    # Option-pricing helpers used for VPI
    # ------------------------------------------------------------------

    def expected_improvement(self, threshold: float) -> float:
        """E[max(Q − threshold, 0)]  (call option above threshold).

        Closed-form for df > 1; numerical quadrature fallback otherwise.
        """
        nu = self.df
        z = (threshold - self.mu) / self.scale

        if nu <= 1.0:
            # Cauchy marginal—use numerical quadrature
            from scipy.integrate import quad  # local import to avoid overhead
            result, _ = quad(
                lambda q: max(q - threshold, 0.0)
                          * stats.t.pdf(q, df=nu, loc=self.mu, scale=self.scale),
                threshold,
                threshold + 50.0 * self.scale,
            )
            return float(result)

        # Closed form: (μ − θ)(1 − F(z)) + σ · ν/(ν−1) · f(z) · (1 + z²/ν)
        survival = float(1.0 - stats.t.cdf(z, df=nu))
        pdf_z = float(stats.t.pdf(z, df=nu))
        tail_moment = self.scale * nu / (nu - 1.0) * pdf_z * (1.0 + z ** 2 / nu)
        return float((self.mu - threshold) * survival + tail_moment)

    def expected_shortfall(self, threshold: float) -> float:
        """E[max(threshold − Q, 0)]  (put option below threshold).

        Closed-form for df > 1; numerical quadrature fallback otherwise.
        """
        nu = self.df
        z = (threshold - self.mu) / self.scale

        if nu <= 1.0:
            from scipy.integrate import quad
            result, _ = quad(
                lambda q: max(threshold - q, 0.0)
                          * stats.t.pdf(q, df=nu, loc=self.mu, scale=self.scale),
                threshold - 50.0 * self.scale,
                threshold,
            )
            return float(result)

        # Closed form: (θ − μ) F(z) + σ · ν/(ν−1) · f(z) · (1 + z²/ν)
        cdf_z = float(stats.t.cdf(z, df=nu))
        pdf_z = float(stats.t.pdf(z, df=nu))
        tail_moment = self.scale * nu / (nu - 1.0) * pdf_z * (1.0 + z ** 2 / nu)
        return float((threshold - self.mu) * cdf_z + tail_moment)


# ---------------------------------------------------------------------------
# Bayesian Q-learning agent
# ---------------------------------------------------------------------------

class BayesianQLearning:
    """Bayesian Q-learning (Dearden et al., 1998).

    Each Q(s, a) has an independent Normal-Gamma posterior.  At every step:
      1.  Select a = argmax_a  [E[Q(s,a)] + VPI(s,a)].
      2.  Observe (r, s') and form the TD target
              x = r + γ · max_{a'} E[Q(s', a')]          (moment-matching)
      3.  Update the Normal-Gamma posterior for Q(s, a) with x.

    VPI(s, a) is the value of perfect information about Q(s, a):
      • For the greedy action a*:   E[max(Q_2nd − Q(s,a*), 0)]   (put option)
      • For any other action a:     E[max(Q(s,a) − Q_best, 0)]   (call option)
    where Q_best = max_{a'} E[Q(s,a')] and Q_2nd = max_{a'≠a*} E[Q(s,a')].

    This formulation ensures VPI ≥ 0 for every action and converges to zero
    as uncertainty collapses.

    Args:
        n_states:  number of discrete states
        n_actions: number of discrete actions
        gamma:     discount factor
        mu0:       prior mean for every Q(s,a)
        lam0:      prior pseudo-count (higher → stronger prior pull)
        alpha0:    Gamma shape of prior
        beta0:     Gamma rate of prior  (beta0/alpha0 ≈ prior reward variance)
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = 0.99,
        mu0: float = 0.0,
        lam0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma

        prior = NormalGamma(mu=mu0, lam=lam0, alpha=alpha0, beta=beta0)
        self._q: list[list[NormalGamma]] = [
            [prior for _ in range(n_actions)] for _ in range(n_states)
        ]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def q_means(self, state: int) -> np.ndarray:
        return np.array([self._q[state][a].mean for a in range(self.n_actions)])

    def q_params(self, state: int, action: int) -> NormalGamma:
        return self._q[state][action]

    # ------------------------------------------------------------------
    # VPI
    # ------------------------------------------------------------------

    def vpi(self, state: int, action: int) -> float:
        """Value of perfect information for Q(state, action)."""
        means = self.q_means(state)
        best_a = int(np.argmax(means))
        p = self._q[state][action]

        if action == best_a:
            # Gain arises only if true Q(s, a*) is worse than second-best
            competing = np.delete(means, action)
            second_best = float(competing.max()) if len(competing) > 0 else -np.inf
            return p.expected_shortfall(second_best)
        else:
            # Gain arises only if true Q(s, a) exceeds current best
            return p.expected_improvement(means[best_a])

    # ------------------------------------------------------------------
    # Action selection and update
    # ------------------------------------------------------------------

    def select_action(self, state: int) -> int:
        """VPI-based action selection: argmax_a [E[Q(s,a)] + VPI(s,a)]."""
        means = self.q_means(state)
        vpi_bonuses = np.array([self.vpi(state, a) for a in range(self.n_actions)])
        return int(np.argmax(means + vpi_bonuses))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """Bayesian update of the Normal-Gamma posterior for Q(state, action)."""
        if done:
            target = reward
        else:
            target = reward + self.gamma * float(np.max(self.q_means(next_state)))
        self._q[state][action] = self._q[state][action].update(target)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def value(self, state: int) -> float:
        return float(np.max(self.q_means(state)))

    def greedy_action(self, state: int) -> int:
        return int(np.argmax(self.q_means(state)))

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
