# Assignment 2 Policy Iteration and Custom Gym Environment

A custom 5x5 Gymnasium maze with PyGame rendering and tabular Policy Iteration.
Each reset samples a start, goal, and two bad cells without overlap. The solver
plans for that fixed layout; it must be run again after generating a new maze.

## Setup

Use Python 3.10 or later. From this directory, create a virtual environment:

```sh
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS and Linux:

```sh
source .venv/bin/activate
```

Install dependencies:

```sh
python -m pip install -r requirements.txt
```

The project was checked with Python 3.12.14, Gymnasium 1.4.0, NumPy 2.3.5,
PyGame 2.6.1, MoviePy 2.2.1, and pytest 9.2.0.

## Run the experiment

```sh
python run_experiment.py
python run_experiment.py --seed 42
python run_experiment.py --seed 7 --render
```

The first two commands print the layout, policy, convergence count, actual
agent path, reward, and stopping condition. The last command also displays
the episode in a PyGame window and needs a graphical desktop.

Coordinates are `(row, column)`, starting at `(0, 0)` in the top-left corner.
The observation is `[agent_row, agent_column, goal_row, goal_column,
bad1_row, bad1_column, bad2_row, bad2_column]`. Actions are
`0=right`, `1=up`, `2=left`, and `3=down`.

Rewards are exactly +1 for entering the goal, -1 for entering either bad cell,
and 0 otherwise. Goal and bad cells end the episode. Boundary moves stay in
place. A 100-step safety limit truncates episodes that do not finish.

## Record the agent

```sh
python record_video.py
```

This uses Gymnasium's `RecordVideo` wrapper and PyGame RGB frames to record
seeds 7, 42, and 123. It solves each new layout before following its policy.
The three included files are:

| File | Seed | Steps | Total reward |
| --- | ---: | ---: | ---: |
| `videos/rl-video-episode-0.mp4` | 7 | 6 | +1 |
| `videos/rl-video-episode-1.mp4` | 42 | 3 | +1 |
| `videos/rl-video-episode-2.mp4` | 123 | 5 | +1 |

Recordings are saved beside the script in `videos/`. Running the script again
overwrites those episode filenames.

## Tests

```sh
python -m pytest -q
```

The 27 tests cover the Gymnasium API, reset seeding, actions, boundaries,
rewards, termination, truncation, terminal values, rendering, and solver
parameters. A test also checks 2,200 non-terminal starting positions across
seeds 0 to 99 against independent breadth-first-search distances. BFS is used
only to verify the Policy Iteration solution.

## Files and submission

- `grid_maze_env.py`: environment, known transition model, and rendering.
- `policy_iteration.py`: iterative policy evaluation and greedy improvement.
- `run_experiment.py`: solve a layout and execute the resulting policy.
- `record_video.py`: record three complete episodes.
- `REPORT.md`: outcomes and answers to all seven assignment questions.
- `test_grid_maze.py`: correctness checks.
- `requirements.txt`: dependencies.
- `videos/`: recorded episodes.

The assignment also requires a GitHub repository. The original project points
to [this repository](https://github.com/VENAMONIX/policy_iteration_custom_gym_environment).
Upload or commit the final source, report, and videos there before submission;
the ZIP does not update the remote repository. Keep local virtual environments
and Python caches out of the repository.
