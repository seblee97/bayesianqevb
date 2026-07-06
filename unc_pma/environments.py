from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Set, Tuple

import numpy as np


class Environment(ABC):
    """Abstract base class for discrete-state, discrete-action RL environments."""

    @abstractmethod
    def reset(self) -> int:
        """Reset to the initial state and return it."""
        ...

    @abstractmethod
    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """Apply action; return (next_state, reward, done, info)."""
        ...

    @property
    @abstractmethod
    def n_states(self) -> int:
        ...

    @property
    @abstractmethod
    def n_actions(self) -> int:
        ...


# ---------------------------------------------------------------------------
# Noisy bandit
# ---------------------------------------------------------------------------

class NoisyBandit(Environment):
    """K-armed bandit where arm k yields reward ~ N(mean_k, std_k²).

    The environment has a single state (0) and never terminates, so it is
    suitable for studying pure exploration vs. exploitation trade-offs.
    """

    def __init__(self, means: np.ndarray, stds: np.ndarray):
        self._means = np.asarray(means, dtype=float)
        self._stds = np.asarray(stds, dtype=float)
        if self._means.shape != self._stds.shape:
            raise ValueError("means and stds must have the same shape")

    def reset(self) -> int:
        return 0

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        reward = float(np.random.normal(self._means[action], self._stds[action]))
        return 0, reward, False, {}

    @property
    def n_states(self) -> int:
        return 1

    @property
    def n_actions(self) -> int:
        return len(self._means)

    @property
    def optimal_action(self) -> int:
        return int(np.argmax(self._means))

    @property
    def means(self) -> np.ndarray:
        return self._means.copy()


# ---------------------------------------------------------------------------
# Noisy gridworld
# ---------------------------------------------------------------------------

class NoisyGridworld(Environment):
    """Grid MDP with stochastic transitions and Gaussian reward noise.

    Coordinate convention: (row, col), origin at top-left.
    Actions: 0=up, 1=right, 2=down, 3=left.

    With probability slip_prob the agent executes a uniformly random action
    instead of the intended one (transition noise).  Goal cells are terminal;
    all other transitions cost step_cost plus additive Gaussian noise.
    Wall cells are impassable—the agent stays in place on collision.

    Args:
        height, width:   grid dimensions
        start:           (row, col) of the starting cell
        goals:           mapping {(row, col): reward} for terminal cells
        walls:           set of impassable (row, col) cells
        slip_prob:       probability of executing a random action
        reward_noise:    std-dev of additive Gaussian noise on every reward
        step_cost:       constant subtracted from non-goal rewards
    """

    _DELTAS: list[Tuple[int, int]] = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def __init__(
        self,
        height: int,
        width: int,
        start: Tuple[int, int],
        goals: Dict[Tuple[int, int], float],
        walls: Optional[Set[Tuple[int, int]]] = None,
        slip_prob: float = 0.1,
        reward_noise: float = 0.1,
        step_cost: float = 0.01,
    ):
        self.height = height
        self.width = width
        self.start = start
        self.goals = dict(goals)
        self.walls = set(walls) if walls else set()
        self.slip_prob = slip_prob
        self.reward_noise = reward_noise
        self.step_cost = step_cost
        self._pos: Tuple[int, int] = start

    # ------------------------------------------------------------------
    def reset(self) -> int:
        self._pos = self.start
        return self._encode(*self._pos)

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        if np.random.random() < self.slip_prob:
            action = int(np.random.randint(4))

        dr, dc = self._DELTAS[action]
        r, c = self._pos
        nr, nc = r + dr, c + dc

        if (0 <= nr < self.height and 0 <= nc < self.width
                and (nr, nc) not in self.walls):
            self._pos = (nr, nc)

        done = self._pos in self.goals
        base_reward = self.goals.get(self._pos, -self.step_cost)
        noise = float(np.random.normal(0.0, self.reward_noise)) if self.reward_noise > 0 else 0.0
        reward = base_reward + noise

        return self._encode(*self._pos), reward, done, {"pos": self._pos}

    # ------------------------------------------------------------------
    @property
    def n_states(self) -> int:
        return self.height * self.width

    @property
    def n_actions(self) -> int:
        return 4

    # ------------------------------------------------------------------
    def _encode(self, r: int, c: int) -> int:
        return r * self.width + c

    def _decode(self, s: int) -> Tuple[int, int]:
        return divmod(s, self.width)

    def render(self) -> str:
        """Return a simple ASCII rendering of the current grid state."""
        lines = []
        for r in range(self.height):
            row = []
            for c in range(self.width):
                pos = (r, c)
                if pos == self._pos:
                    row.append("A")
                elif pos in self.walls:
                    row.append("#")
                elif pos in self.goals:
                    row.append("G" if self.goals[pos] > 0 else "T")
                elif pos == self.start:
                    row.append("S")
                else:
                    row.append(".")
            lines.append(" ".join(row))
        return "\n".join(lines)
