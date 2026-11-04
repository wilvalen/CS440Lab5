---
name: qlearning-experiment
description: Run and interpret Q-learning experiments for the Lab 4 vacuum robot. Use when asked to train the agent, try or compare hyperparameters (living reward, discount factor gamma, learning rate alpha, epsilon), run the hyperparameter sweep, or explain how a hyperparameter changes the learned policy.
argument-hint: "[--living-reward R] [--gamma G] [--alpha A] [--epsilon E] [--episodes N] | sweep"
allowed-tools: Bash(python3*), Bash(/opt/anaconda3/bin/python3*), Read
---

# Q-learning experiment (Lab 4)

The world is the 4x4 grid from the lab: `start` at the top left, `X` at (1,1), `-1` at (3,1), `+10` at (3,3).
The agent (`QLearningAgent.py`) learns **only** from `environment.step()`. It never sees T or R.
`reference.py` (value iteration with the true T and R) is used **only to grade** the result.

## Interpreter
`aima/mdp.py` imports numpy. Use a Python that has it: `/opt/anaconda3/bin/python3.13`
(or `$LAB_PYTHON` if set). Plain `/Library/.../python3` (3.11) has no numpy and will fail.

## Steps

1. **Pick the mode from `$ARGUMENTS`:**
   - empty: train with the selected values: `python3.13 run_QLearning.py --no-animate`
   - flags such as `--alpha 0.3 --epsilon 0.05`: pass them through to `run_QLearning.py --no-animate`
   - `sweep`: run `python3.13 hyperparameter_sweep.py` (~20 s; add `--quick` if asked for speed)
   - To compare a candidate with the selected values, run both with the **same `--seed`**, then also with seeds 1-4. A single seed is noisy.

2. **Report** in this format (keep it short):
   - Hyperparameters used
   - Learning curve: did the goal rate rise and level off?
   - Learned policy grid next to the optimal policy grid; name any cells that differ
   - Metrics: % of states with an optimal action, V(start) learned vs. true, greedy goal / pit rate
   - One sentence on *why*, using the guide below

3. **Do not change `SELECTED` in `run_QLearning.py`** unless the user asks. If they do, also update
   the table in `README.md` and rerun `python3.13 -m unittest test_qlearning`.

## Interpretation guide
- **living reward**: with gamma=0.9 the optimal policy stays the same from 0 to -1.5. At > 0 the agent
  wanders instead of finishing. At <= -2 it jumps into the -1 pit from (3,0) to stop paying the step cost.
- **gamma**: low gamma gives short-sighted values that converge fast. Near 1, values spread slowly and
  need many more episodes.
- **alpha**: the world is random (20% slip), so a large alpha follows the newest sample and the values
  jump around. A tiny alpha learns too slowly in 3000 episodes.
- **epsilon**: 0 relies only on the optimistic start (Q=0 > negative returns) and leaves cells off
  the main path untried. Higher values fill in more of the table but lower the reward collected
  during training.
- "% optimal" counts every cell, including ones the robot rarely visits from `start`. The greedy goal rate
  is the measure that matters for the robot.
