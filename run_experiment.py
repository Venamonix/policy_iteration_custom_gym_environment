"""Run Policy Iteration on one generated maze and print its outcome."""

from __future__ import annotations

import argparse

from grid_maze_env import GridMazeEnv
from policy_iteration import PolicyIteration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--render", action="store_true", help="Show the agent in a PyGame window.")
    args = parser.parse_args()
    env = GridMazeEnv(render_mode="human" if args.render else None)
    try:
        observation, info = env.reset(seed=args.seed)
        result = PolicyIteration(env).solve()
        print("Grid:", env.grid_size, "x", env.grid_size, "| Seed:", args.seed)
        print("Coordinates: (row, column), top-left = (0, 0)")
        print("Agent:", info["agent"], "Goal:", info["goal"], "Bad cells:", info["bad_cells"])
        print("Policy-iteration iterations:", result.iterations)
        print("Policy (R=right, U=up, L=left, D=down; G/X=terminal):")
        for row in range(env.grid_size):
            symbols = []
            for column in range(env.grid_size):
                position = (row, column)
                if position == env.goal_position:
                    symbol = "G"
                elif position in env.bad_positions:
                    symbol = "X"
                else:
                    symbol = "RULD"[result.policy[row * env.grid_size + column]]
                symbols.append(symbol)
            print(" ".join(symbols))

        path = [env.agent_position]
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            state = int(observation[0] * env.grid_size + observation[1])
            observation, reward, terminated, truncated, _ = env.step(int(result.policy[state]))
            total_reward += reward
            path.append(env.agent_position)
        print("Path:", " -> ".join(map(str, path)))
        print("Steps:", len(path) - 1, "| Total reward:", total_reward)
        print("Terminated:", terminated, "| Truncated:", truncated)
    finally:
        env.close()


if __name__ == "__main__":
    main()
