from __future__ import annotations

import numpy as np

from unc_pma.replay_buffer import Experience


class UniformPriority:
    def __call__(self, exp: Experience) -> float:
        return 1.0


class TDErrorPriority:
    """Priority = |TD error| = |r + gamma * max_a' E[Q(s',a')] - E[Q(s,a)]|."""

    def __init__(self, agent, gamma: float = 0.0):
        self.agent = agent
        self.gamma = gamma

    def __call__(self, exp: Experience) -> float:
        if exp.done or self.gamma == 0.0:
            target = exp.reward
        else:
            target = exp.reward + self.gamma * float(
                np.max(self.agent.q_means(exp.next_state)))
        return abs(target - self.agent.q_params(exp.state, exp.action).mean)


class UncertaintyPriority:
    """Priority = posterior scale of Q(s, a)  (epistemic uncertainty)."""

    def __init__(self, agent):
        self.agent = agent

    def __call__(self, exp: Experience) -> float:
        return self.agent.q_params(exp.state, exp.action).scale


class VPIPriority:
    """Priority = VPI bonus of the (s, a) pair  (Dearden et al. 1998).

    The posterior marginal Q(s,a) ~ t(df, mu, scale) gives closed-form
    option prices via NormalGamma.expected_improvement / expected_shortfall:

      greedy arm a*:   E[max(Q_2nd − Q(s,a*), 0)]   put  — gain if a* is worse than 2nd-best
      any other arm a: E[max(Q(s,a)  − Q_best, 0)]  call — gain if a  beats current best

    VPI ≥ 0 always; collapses to 0 as the posterior concentrates.
    Works for any agent that exposes q_means(s) and q_params(s, a).
    """

    def __init__(self, agent):
        self.agent = agent

    def __call__(self, exp: Experience) -> float:
        s, a   = exp.state, exp.action
        means  = self.agent.q_means(s)
        best_a = int(np.argmax(means))
        p      = self.agent.q_params(s, a)

        if a == best_a:
            competing   = np.delete(means, a)
            second_best = float(competing.max()) if len(competing) > 0 else -np.inf
            return p.expected_shortfall(second_best)
        else:
            return p.expected_improvement(float(means[best_a]))


class EVBPriority:
    """Expected Value of Backup (Mattar & Daw 2018): GAIN × NEED.

    GAIN = max(0, V(s) after predicted conjugate update − V(s) before).
    NEED = 1 everywhere (approximation; proper NEED is the on-policy successor
    representation, which would weight states by expected future visit count).
    """

    def __init__(self, agent, gamma: float = 0.0):
        self.agent = agent
        self.gamma = gamma

    def __call__(self, exp: Experience) -> float:
        s, a, r, s_next, done = exp
        means    = self.agent.q_means(s)
        v_before = float(np.max(means))
        T = r if (done or self.gamma == 0.0) else (
            r + self.gamma * float(np.max(self.agent.q_means(s_next))))
        ng             = self.agent.q_params(s, a)
        mu_after       = (ng.lam * ng.mu + T) / (ng.lam + 1.0)
        means_after    = means.copy()
        means_after[a] = mu_after
        return max(0.0, float(np.max(means_after)) - v_before)
