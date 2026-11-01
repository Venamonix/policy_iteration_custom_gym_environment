"""Check the Gym API, maze rules, and policies against independent BFS distances."""

from collections import deque

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from grid_maze_env import GridMazeEnv
from policy_iteration import PolicyIteration


@pytest.fixture
def env():
    maze = GridMazeEnv()
    maze.reset(seed=0)
    maze._agent = (2, 2)
    maze._goal = (4, 4)
    maze._bad_cells = ((0, 0), (0, 1))
    yield maze
    maze.close()


def test_gymnasium_api():
    maze = GridMazeEnv()
    try:
        check_env(maze, skip_render_check=True)
    finally:
        maze.close()


def test_reset_observation_spaces_and_reproducible_seed(env):
    first, _ = env.reset(seed=7)
    second, _ = env.reset(seed=7)
    assert np.array_equal(first, second)
    assert first.shape == (8,)
    assert env.observation_space.contains(first)
    assert len({tuple(first[i:i + 2]) for i in range(0, 8, 2)}) == 4
    layouts = {tuple(env.reset(seed=seed)[0]) for seed in range(10)}
    assert len(layouts) > 1


@pytest.mark.parametrize("action, expected", [
    (GridMazeEnv.RIGHT, (2, 3)),
    (GridMazeEnv.UP, (1, 2)),
    (GridMazeEnv.LEFT, (2, 1)),
    (GridMazeEnv.DOWN, (3, 2)),
])
def test_action_directions(env, action, expected):
    observation, reward, terminated, truncated, _ = env.step(action)
    assert tuple(observation[:2]) == expected
    assert reward == 0.0
    assert not terminated and not truncated


@pytest.mark.parametrize("position, action", [
    ((0, 3), GridMazeEnv.UP),
    ((4, 3), GridMazeEnv.DOWN),
    ((2, 0), GridMazeEnv.LEFT),
    ((2, 4), GridMazeEnv.RIGHT),
])
def test_boundary_move_stays_in_place(env, position, action):
    env._agent = position
    observation, reward, terminated, truncated, _ = env.step(action)
    assert tuple(observation[:2]) == position
    assert reward == 0.0
    assert not terminated and not truncated


@pytest.mark.parametrize("position, action, reward, terminated", [
    ((4, 3), GridMazeEnv.RIGHT, 1.0, True),
    ((1, 0), GridMazeEnv.UP, -1.0, True),
    ((2, 2), GridMazeEnv.RIGHT, 0.0, False),
])
def test_rewards_and_episode_endings(env, position, action, reward, terminated):
    env._agent = position
    _, actual_reward, actual_terminated, truncated, _ = env.step(action)
    assert actual_reward == reward
    assert actual_terminated is terminated
    assert not truncated


def test_time_limit_truncates_without_terminal_reward(env):
    env.max_steps = 1
    _, reward, terminated, truncated, _ = env.step(GridMazeEnv.RIGHT)
    assert reward == 0.0
    assert not terminated and truncated


def test_goal_on_last_allowed_step_is_success(env):
    env.max_steps = 1
    env._agent = (4, 3)
    _, reward, terminated, truncated, _ = env.step(GridMazeEnv.RIGHT)
    assert reward == 1.0
    assert terminated and not truncated


def test_terminal_model_is_absorbing_and_does_not_mutate_env(env):
    before = env._get_obs().copy()
    for position in (env.goal_position, *env.bad_positions):
        for action in range(4):
            assert env.transition(position, action) == (position, 0.0, True)
    assert np.array_equal(before, env._get_obs())
    assert env._steps == 0


def test_terminal_values_are_zero(env):
    result = PolicyIteration(env).solve()
    for row, column in (env.goal_position, *env.bad_positions):
        assert result.values[row * env.grid_size + column] == 0.0


def goal_distances(env):
    """Independent shortest paths; BFS is a test oracle, not the RL solver."""
    distances = {env.goal_position: 0}
    queue = deque([env.goal_position])
    while queue:
        row, column = queue.popleft()
        for neighbor in ((row + 1, column), (row - 1, column),
                         (row, column + 1), (row, column - 1)):
            if (0 <= neighbor[0] < env.grid_size
                    and 0 <= neighbor[1] < env.grid_size
                    and neighbor not in env.bad_positions
                    and neighbor not in distances):
                distances[neighbor] = distances[(row, column)] + 1
                queue.append(neighbor)
    return distances


def test_optimal_values_and_rollouts_for_100_layouts(env):
    # Each layout has 22 possible non-terminal starts: 2,200 cases in total.
    for seed in range(100):
        env.reset(seed=seed)
        result = PolicyIteration(env).solve()
        distances = goal_distances(env)
        assert result.policy.shape == (25,)
        assert all(env.action_space.contains(int(a)) for a in result.policy)
        for row in range(5):
            for column in range(5):
                start = (row, column)
                if env.is_terminal(start):
                    continue
                state = row * 5 + column
                distance = distances.get(start)
                expected_value = 0.99 ** (distance - 1) if distance is not None else 0.0
                assert result.values[state] == pytest.approx(expected_value, abs=1e-8)
                env._agent, env._steps = start, 0
                total_reward = 0.0
                for steps in range(1, env.max_steps + 1):
                    state = env.agent_position[0] * 5 + env.agent_position[1]
                    _, reward, terminated, truncated, _ = env.step(int(result.policy[state]))
                    total_reward += reward
                    if terminated or truncated:
                        break
                if distance is not None:
                    assert env.agent_position == env.goal_position
                    assert steps == distance
                    assert total_reward == 1.0
                    assert terminated and not truncated
                else:
                    assert total_reward == 0.0
                    assert truncated and not terminated


@pytest.mark.parametrize("start, goal", [((4, 4), (0, 0)), ((0, 0), (4, 4))])
def test_unreachable_goal_prefers_zero_reward_to_bad_cell(env, start, goal):
    env._agent, env._goal = start, goal
    env._bad_cells = ((0, 1), (1, 0))
    result = PolicyIteration(env).solve()
    state = start[0] * 5 + start[1]
    assert result.values[state] == pytest.approx(0.0, abs=1e-8)
    for _ in range(env.max_steps):
        state = env.agent_position[0] * 5 + env.agent_position[1]
        _, reward, terminated, truncated, _ = env.step(int(result.policy[state]))
        assert reward == 0.0 and not terminated
        if truncated:
            break
    assert truncated


@pytest.mark.parametrize("kwargs", [
    {"gamma": -0.1}, {"gamma": 1.0}, {"gamma": float("nan")},
    {"theta": 0.0}, {"theta": -1.0}, {"theta": float("inf")},
])
def test_invalid_solver_parameters(env, kwargs):
    with pytest.raises(ValueError):
        PolicyIteration(env, **kwargs)


def test_rgb_rendering_returns_changing_uint8_frames():
    maze = GridMazeEnv(render_mode="rgb_array")
    try:
        maze.reset(seed=7)
        before = maze.render()
        maze.step(GridMazeEnv.UP)
        after = maze.render()
        assert before.dtype == np.uint8
        assert before.ndim == 3 and before.shape[2] == 3
        assert after.shape == before.shape
        assert not np.array_equal(before, after)
    finally:
        maze.close()
