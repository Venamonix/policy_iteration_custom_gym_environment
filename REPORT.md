# Assignment 2 Report

## Outcome summary

The custom Gymnasium environment and Policy Iteration solver complete the three
recorded 5x5 mazes successfully, with total reward +1 in every episode. For the
main reproducible example, seed 7, the policy becomes stable after **4 outer
iterations** and reaches the goal in **6 actions** without entering a bad cell.

| Seed | Start | Goal | Bad cells | Policy iterations | Actions to goal | Reward |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 7 | (4, 0) | (2, 4) | (4, 2), (3, 1) | 4 | 6 | +1 |
| 42 | (2, 0) | (0, 1) | (3, 0), (3, 2) | 8 | 3 | +1 |
| 123 | (3, 0) | (2, 4) | (0, 0), (0, 1) | 4 | 5 | +1 |

The path for seed 7 is:

`(4,0) -> (3,0) -> (2,0) -> (2,1) -> (2,2) -> (2,3) -> (2,4)`

These are planning results from a known model. No training episodes are needed
to estimate transitions. A policy is calculated separately for each new layout.

## Environment and model

`GridMazeEnv` inherits from `gymnasium.Env`. At each reset it randomly samples
four different cells without replacement: the agent start, one goal, and two
bad cells. Positions remain fixed during that episode except for the agent.
The observation is an eight-integer `MultiDiscrete([5] * 8)` vector:

`[agent_row, agent_column, goal_row, goal_column, bad1_row, bad1_column, bad2_row, bad2_column]`

Coordinates use `(row, column)`, with `(0, 0)` at the top left. This convention
is used consistently for all four coordinate pairs. The four discrete actions
are `0=right`, `1=up`, `2=left`, and `3=down`. Movement is deterministic and
an attempted move beyond the boundary leaves the agent in its current cell.

The assignment's rewards are preserved: +1 on entering the goal, -1 on
entering either bad cell, and 0 otherwise. Reaching the goal or a bad cell
terminates the episode. These cells are modeled as absorbing terminal states
with zero future value; the entry reward is paid only once. A separate 100-step
safety limit truncates unfinished episodes and does not add a reward penalty.
The assignment does not explicitly specify termination or boundary behavior,
so these choices define them for this implementation.

Both `step()` and the solver use the environment's pure `transition()` method.
This keeps the planning model consistent with the executed environment.
PyGame draws the grid, start, agent, goal, bad cells, and episode status.
The same renderer supports a human window and RGB arrays for `RecordVideo`.

## Policy Iteration method

For one fixed layout, the solver indexes agent positions by
`state = row * 5 + column`. It starts with the action "right" in every state
and zero initial values. The discount is `gamma = 0.99` and the evaluation
tolerance is `theta = 1e-10`.

1. **Policy evaluation:** repeatedly sweep through the 25 states. For the
   current action, compute `V(s) = r + gamma * V(next_state)` for a non-terminal
   transition, or `V(s) = r` when that transition terminates. Updates are made
   in place, and evaluation stops when the largest change in a sweep is below
   `theta`. A state's value after the episode has already ended is zero.
2. **Policy improvement:** calculate the same one-step return for all four
   actions and choose the largest. `numpy.argmax` breaks exact ties by choosing
   the first action in the order right, up, left, down.
3. Repeat until a complete improvement pass changes no actions.

With zero intermediate rewards and `gamma < 1`, a shorter safe route gives a
larger discounted return than a longer safe route. If the goal is unreachable,
remaining in safe cells gives return 0 and is preferable to the -1 bad-cell
reward. Random sampling can isolate a corner behind two bad cells, so not every
possible maze must have a successful route. Such an episode reaches the safety
limit rather than being forced into a bad cell.

## Questions

### 1 State space size

The answer depends on whether the layout is held fixed or included in the state:

| Interpretation | Size | Explanation |
| --- | ---: | --- |
| Agent positions in one fixed 5x5 layout | **25** | 22 non-terminal positions and 3 terminal cells; this is the solver's table. |
| Nominal eight-coordinate observation space | **390,625** | `5^8 = 25^4`; includes overlapping goal and bad-cell placements that reset never generates. |
| Valid initial observations with four distinct cells | **303,600** | `25 * 24 * 23 * 22`; the bad cells occupy separate labeled slots in the observation. |
| Coordinate configurations allowing terminal agent overlaps | **345,000** | `25 * 24 * 23 * 25`; goal and bad cells stay distinct, while the agent may occupy any cell, including a terminal cell. |

The last count describes structurally valid coordinate configurations, rather
than claiming that every one is reachable in every layout. The 303,600 count is
for reset observations, not the entire space after agent movement. Although
the environment returns the full eight-coordinate observation, a 25-state table
is sufficient while the goal and bad-cell positions are fixed. That table must
be recomputed when the layout changes.

### 2 Optimizing Policy Iteration

This implementation uses a 25-state array for the current layout, takes advantage
of a single deterministic successor for each action, updates values in place,
and stops evaluation at a numerical tolerance. It does not enumerate every
possible maze configuration.

Further improvements for a larger maze would be to precompute the successor,
reward, and termination arrays once per layout; skip terminal states during
evaluation and improvement; and restrict planning to relevant reachable states.
Modified Policy Iteration can use a limited number of evaluation sweeps before
improving the policy. Sparse transition storage also avoids allocating a dense
state-to-state probability matrix. These are possible extensions, rather than
extra algorithms required for this 25-state assignment.

### 3 Iterations until a stable policy

For seed 7 the answer is **4 outer policy-improvement iterations**, including
the final pass that confirms stability. Seeds 42 and 123 require 8 and 4,
respectively. These counts are measured by the included scripts. They are not
the number of agent actions or the number of inner value-evaluation sweeps.
Counts can change with the layout, initial policy, discount, tolerance, update
order, and tie-breaking rule; there is no single count for all random mazes.

### 4 Multiple goal cells

In a hypothetical extension, both goals would be terminal states, and entering
either would give its goal reward. Policy Iteration would maximize the expected
discounted return, which usually selects the nearer goal when rewards are equal
and movement is deterministic.

For example, let the agent start at `(2, 2)`, with equal +1 goals at `(2, 4)`
and `(0, 0)` and no bad cells blocking the relevant routes. The first goal takes
2 actions, giving return `0.99^(2-1) = 0.99`. The second takes 4 actions, giving
`0.99^(4-1) = 0.970299`. The policy therefore prefers the goal at `(2, 4)`.
Bad cells can change which route is shortest. Different goal rewards can make
a farther goal preferable, and equal action returns are resolved consistently
by the action ordering. The submitted environment retains the assignment's
single randomly generated goal; this question concerns an extension.

### 5 A 100x100 maze

Yes, classical Policy Iteration can work if the maze is discrete and the model
is known. A fixed 100x100 layout has 10,000 agent-position states and four
actions, so one deterministic evaluation sweep or improvement pass examines
roughly 40,000 state-action computations or fewer. Several sweeps and outer
iterations may be needed, making computation greater than for 25 states.

Storage depends on the implementation. Deterministic successor arrays take
space proportional to `states * actions`, whereas a dense transition tensor
would take `states * actions * states` entries. Keeping all possible goal and
bad-cell placements in one global table causes a much larger state-space
explosion, approaching `10,000^4 = 10^16` coordinate combinations. A fixed-layout
solver with sparse or deterministic transitions is much more practical. The
submitted environment stays 5x5 as required; this answer is a scaling analysis.

### 6 A continuous-space maze

Classical **tabular** Policy Iteration cannot be applied directly to every
position in a continuous maze because there are infinitely many states to
enumerate and store. A known transition model alone does not remove that
problem. Discretizing the positions gives a finite approximation, with a
tradeoff between accuracy and computational cost. Approximate policy iteration
or continuous-state methods using function approximation, including suitable
actor-critic methods, are alternatives. Continuous states do not necessarily
imply continuous actions.

### 7 Moving bad cells

Yes in principle, if the state contains enough information to be Markov and the
bad-cell movement has a known, stationary transition model. A state would include
the agent, goal, and both current ghost positions. If ghost movement also depends
on direction or a movement phase, that information must be included too.

For example, if each ghost moves uniformly among its legal neighboring cells,
the transition model must enumerate the joint movement outcomes and their
probabilities. Policy evaluation then sums over these outcomes instead of using
one deterministic successor. The order of movement and collision rules also
need to be defined. Planning only from the agent's position would miss hazards
that depend on the ghosts' current locations.

The enlarged state space makes exact planning much more expensive. If the
movement model is unknown, ordinary model-based Policy Iteration cannot be used
directly without learning a model. If the rules change over time, a single fixed
stationary model is insufficient unless the relevant time or phase is represented
in the state. The current static 25-state model would therefore need to change.

## Verification and reproduction

Run `python run_experiment.py` for seed 7 and `python record_video.py` for all
three recordings. The videos use `RecordVideo`, render at 2 frames per second,
include the initial and terminal frames, and show 6, 3, and 5 agent actions.
All three episodes terminate at their goals without truncation.

`python -m pytest -q` passes **27 tests**. The checks cover Gymnasium's API,
seed reproducibility, all four action directions and boundaries, rewards,
terminal values, time limits, and RGB rendering. For seeds 0 through 99, all
22 non-terminal starting positions per layout are checked: **2,200 rollout
cases**. Independently computed BFS distances verify shortest safe routes and
the expected value `gamma^(distance-1)`. Separate cases check a trapped start
and an isolated goal. BFS is used only as a test reference; the submitted
solver itself performs Policy Iteration.

## References

- [Gymnasium environment creation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
- [Gymnasium recording agents](https://gymnasium.farama.org/introduction/record_agent/)
- [Gymnasium RecordVideo API](https://gymnasium.farama.org/api/wrappers/misc_wrappers/)
- The assignment's Policy Iteration notes link: [Mastering Reinforcement Learning](https://gibberblot.github.io/rl-notes/single-agent/policy-iteration.html)
