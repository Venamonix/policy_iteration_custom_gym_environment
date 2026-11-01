"""Record three solved mazes with Gymnasium's RecordVideo wrapper."""

from __future__ import annotations

import os
from pathlib import Path

from gymnasium.wrappers import RecordVideo

from grid_maze_env import GridMazeEnv
from policy_iteration import PolicyIteration


def main() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    video_directory = Path(__file__).resolve().parent / "videos"
    video_directory.mkdir(exist_ok=True)
    base_env = GridMazeEnv(render_mode="rgb_array")
    env = RecordVideo(base_env, video_folder=str(video_directory), episode_trigger=lambda episode: True)
    try:
        for seed in (7, 42, 123):
            observation, _ = env.reset(seed=seed)
            result = PolicyIteration(base_env).solve()
            terminated = truncated = False
            steps, total_reward = 0, 0.0
            while not (terminated or truncated):
                state = int(observation[0] * base_env.grid_size + observation[1])
                observation, reward, terminated, truncated, _ = env.step(int(result.policy[state]))
                steps += 1
                total_reward += reward
            print(f"Seed {seed}: {result.iterations} policy rounds, {steps} steps, reward {total_reward:+.0f}")
    finally:
        env.close()
    print(f"Video written to {video_directory.resolve()}")


if __name__ == "__main__":
    main()
