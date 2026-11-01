"""Tabular Policy Iteration for a fixed Grid Maze layout."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grid_maze_env import GridMazeEnv


@dataclass
class PolicyIterationResult:
    policy: np.ndarray
    values: np.ndarray
    iterations: int


class PolicyIteration:
    """Solve the current deterministic maze using policy iteration."""

    def __init__(self, env: GridMazeEnv, gamma: float = 0.99, theta: float = 1e-10) -> None:
        if not np.isfinite(gamma) or not 0 <= gamma < 1:
            raise ValueError("gamma must be in [0, 1) for discounted policy evaluation.")
        if not np.isfinite(theta) or theta <= 0:
            raise ValueError("theta must be finite and positive.")
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.n_states = env.grid_size * env.grid_size
        self.n_actions = env.action_space.n

    def _position(self, state: int) -> tuple[int, int]:
        return divmod(state, self.env.grid_size)

    def _state(self, position: tuple[int, int]) -> int:
        return position[0] * self.env.grid_size + position[1]

    def _transition(self, state: int, action: int) -> tuple[int, float, bool]:
        next_position, reward, terminated = self.env.transition(self._position(state), action)
        return self._state(next_position), reward, terminated

    def _evaluate(self, policy: np.ndarray, values: np.ndarray) -> None:
        while True:
            delta = 0.0
            for state in range(self.n_states):
                action = int(policy[state])
                next_state, reward, terminated = self._transition(state, action)
                updated = reward if terminated else reward + self.gamma * values[next_state]
                delta = max(delta, abs(updated - values[state]))
                values[state] = updated
            if delta < self.theta:
                return

    def solve(self) -> PolicyIterationResult:
        # Reset the environment before solving; its layout must stay fixed here.
        policy = np.zeros(self.n_states, dtype=np.int64)
        values = np.zeros(self.n_states, dtype=np.float64)
        iterations = 0

        while True:
            iterations += 1
            self._evaluate(policy, values)
            stable = True
            for state in range(self.n_states):
                old_action = int(policy[state])
                action_values = np.empty(self.n_actions, dtype=np.float64)
                for action in range(self.n_actions):
                    next_state, reward, terminated = self._transition(state, action)
                    action_values[action] = reward if terminated else reward + self.gamma * values[next_state]
                policy[state] = int(np.argmax(action_values))
                stable &= old_action == policy[state]
            if stable:
                return PolicyIterationResult(policy, values, iterations)
