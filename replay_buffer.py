from __future__ import annotations

from collections import deque
from typing import Callable, NamedTuple, Protocol, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bayesian_qlearning import BayesianQLearning


class Experience(NamedTuple):
    state: int
    action: int
    reward: float
    next_state: int
    done: bool


class ReplayBuffer:
    """Circular replay buffer storing (s, a, r, s', done) transitions.

    Args:
        capacity: maximum number of transitions stored; oldest are evicted first.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._buf: deque[Experience] = deque(maxlen=capacity)
        self.capacity = capacity

    def push(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        self._buf.append(Experience(state, action, float(reward), next_state, bool(done)))

    def sample(self, n: int) -> list[Experience]:
        """Return n experiences drawn uniformly without replacement (or with, if n > len)."""
        buf = list(self._buf)
        replace = n > len(buf)
        indices = np.random.choice(len(buf), size=n, replace=replace)
        return [buf[i] for i in indices]

    def replay_into(self, agent: "BayesianQLearning", n: int) -> None:
        """Sample n experiences and call agent.update on each."""
        for exp in self.sample(n):
            agent.update(exp.state, exp.action, exp.reward, exp.next_state, exp.done)

    def __len__(self) -> int:
        return len(self._buf)

    def __repr__(self) -> str:
        return f"ReplayBuffer(capacity={self.capacity}, size={len(self)})"


# ---------------------------------------------------------------------------
# Priority protocol
# ---------------------------------------------------------------------------

class PriorityFn(Protocol):
    """Any callable (Experience) -> float qualifies as a priority function."""
    def __call__(self, exp: Experience) -> float: ...


# ---------------------------------------------------------------------------
# Prioritized replay buffer
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """Replay buffer that samples experiences proportional to a priority function.

    The priority function is a plain callable ``(Experience) -> float`` supplied
    at construction, making it trivial to slot in different schemes (TD-error,
    posterior uncertainty, VPI, recency, …).  Agent-dependent callables should
    close over the agent reference.

    Sampling probability for experience i:
        P(i) ∝ (priority_fn(exp_i) + epsilon) ** alpha

    Args:
        capacity:    maximum number of transitions; oldest are evicted first.
        priority_fn: ``(Experience) -> float``; evaluated once at push time.
        alpha:       priority exponent — 0 recovers uniform sampling, 1 gives
                     fully proportional sampling.
        epsilon:     small constant added to every raw priority so that every
                     transition remains sample-able regardless of its score.
    """

    def __init__(
        self,
        capacity: int,
        priority_fn: Callable[[Experience], float],
        alpha: float = 1.0,
        epsilon: float = 1e-6,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self.priority_fn = priority_fn
        self.alpha = alpha
        self.epsilon = epsilon
        self._buf: deque[Experience] = deque(maxlen=capacity)
        self._priorities: deque[float] = deque(maxlen=capacity)

    def push(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        exp = Experience(state, action, float(reward), next_state, bool(done))
        p = max(float(self.priority_fn(exp)), 0.0) + self.epsilon
        self._buf.append(exp)
        self._priorities.append(p)

    def sample(self, n: int) -> list[Experience]:
        """Sample n experiences with probability proportional to priority^alpha."""
        priorities = np.array(self._priorities, dtype=float) ** self.alpha
        probs = priorities / priorities.sum()
        buf = list(self._buf)
        replace = n > len(buf)
        indices = np.random.choice(len(buf), size=n, replace=replace, p=probs)
        return [buf[i] for i in indices]

    def replay_into(self, agent: "BayesianQLearning", n: int) -> None:
        """Sample n experiences and call agent.update on each."""
        for exp in self.sample(n):
            agent.update(exp.state, exp.action, exp.reward, exp.next_state, exp.done)

    def refresh_priorities(self) -> None:
        """Recompute every stored priority using the current state of priority_fn.

        Call this after a batch of agent updates when the priority function
        depends on a changing agent (e.g. TD-error or uncertainty schemes).
        """
        new_p = [max(float(self.priority_fn(exp)), 0.0) + self.epsilon
                 for exp in self._buf]
        self._priorities = deque(new_p, maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def __repr__(self) -> str:
        return (f"PrioritizedReplayBuffer(capacity={self.capacity}, "
                f"size={len(self)}, alpha={self.alpha})")
