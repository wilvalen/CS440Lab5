# CS440 Lab 4: Vacuum Robot with Q-Learning

```
start  0   0   0
  0    X   0   0
  0    0   0   0
  0   -1   0   10
```

In Lab 3 the vacuum used **value iteration**, which needs the transition model T and the reward
function R. In this lab the robot is not given T or R. It uses **Q-learning** (model-free reinforcement
learning): it acts, observes `(s, a, r, s')`, and updates a Q-table:

```
Q(s,a) ← Q(s,a) + α · [ r + γ · max_a' Q(s',a') − Q(s,a) ]      (target = r if s' is terminal)
```

T and R are hidden inside `environment.step()`. The robot moves as intended 80% of the time and
slips sideways 10% each way, the same model as AIMA. Each step pays the living reward. Entering
`+10` or `-1` pays that value and ends the episode.

## Selected hyperparameters

| Hyperparameter | Value | Why |
|---|---|---|
| Living reward | **−0.1** | With γ = 0.9 the optimal policy is the safe route (away from the −1) for any living reward from 0 to −1.5. At > 0 the robot wanders instead of finishing. At ≤ −2 it jumps into the −1 pit from (3,0) to stop paying the step cost. −0.1 sits well inside the safe range: the robot is pushed to finish, but not enough to take risks near the pit. |
| Discount γ | **0.9** | Same as Lab 3, so the results can be compared. +10 reached 6 steps from start is still worth ≈ 4.6 there. γ = 0.99 needs far more episodes before the values spread back to the start (77% optimal vs. 86% at 3000 episodes). |
| Learning rate α | **0.1** | Moves are random, so each Q-value should average many samples. α = 0.1 gave the smallest V(start) error (0.13) and the best goal rate among values that learn fast enough. α ≥ 0.5 follows the newest sample and the values swing (70% optimal). α = 0.01 learns too slowly (V error 0.30). |
| Exploration ε | **0.2** | ε-greedy. Going from 0.1 to 0.2 raised the share of states with the optimal action from 86% to 92% and cut the V(start) error from 0.13 to 0.04, with the same goal rate. Above 0.2 the gains are small (94% at 0.3) and the robot collects less reward while training. ε = 0 relies only on Q starting at 0 and leaves cells off the main path unexplored. |

Training runs 3000 episodes from `start`. The evidence (10 seeds per setting) comes from `python3.13 hyperparameter_sweep.py`.

## Results (seed 0)

```
Learned greedy policy        Optimal (value iteration, true T & R)
>   >    v   v               >   >    v   v
v   X    v   v               v   X    v   v
>   >    >   v               >   >    >   v
^   -1   ^   +10             ^   -1   >   +10
```

* The learned policy is optimal in **92%** of states. V(start) is **4.64** learned vs. **4.59** true.
* The greedy robot reaches **+10 in 97.5%** of 1000 runs (the misses come from random slips).
* The one difference is (3,2), a cell next to the pit that the robot rarely visits from `start`.
  Q-learning only learns well where it has experience.

## Project structure

```
CS440Lab5/
├── README.md                    ← this file
├── .gitignore
│
├── vacuum.py                    base vacuum class (shared by Lab 3 and Lab 4)
├── environment.py               4x4 grid world; Lab 4 adds set_RL_env() / step() (hidden T and R)
├── print_details.py             prints the grid, policy arrows, and Q-value tables
│
├── QLearningAgent.py            ★ Lab 4: Q-learning agent (model-free RL)
├── run_QLearning.py             ★ Lab 4: main driver + SELECTED hyperparameters
├── hyperparameter_sweep.py      ★ Lab 4: evidence for the hyperparameter choices
├── reference.py                 ★ Lab 4: true-model value iteration, used only for grading
├── test_qlearning.py            ★ Lab 4: unit tests (run by the hook)
│
├── MDPAgent.py                  Lab 3: value-iteration agent (model-based baseline)
├── run_MDP.py                   Lab 3: driver
├── aima/                        AIMA library code (mdp.py, utils.py); needs numpy
│
└── .claude/                     Claude Code context engineering
    ├── settings.json            HOOK: PostToolUse → check_python.sh
    ├── hooks/check_python.sh    compile + test after every .py edit
    ├── skills/
    │   └── qlearning-experiment/SKILL.md   SKILL: /qlearning-experiment
    └── agents/
        └── rl-reviewer.md       SUB-AGENT: correctness reviewer
```

How the parts connect: `run_QLearning.py` builds an `environment` and a `QLearningAgent`. The
agent calls `env.step()` thousands of times to fill its Q-table. Afterwards `reference.py` checks the
learned policy against the true optimum.

## Files

| File | Purpose |
|---|---|
| `QLearningAgent.py` | Q-learning agent: ε-greedy, Q-table, TD update, episodes |
| `environment.py` | `set_RL_env()` / `step()` added. The hidden T and R live here |
| `run_QLearning.py` | Main driver: train, print Q-values and policy, animate a run. `SELECTED` hyperparameters |
| `hyperparameter_sweep.py` | Changes one hyperparameter at a time across 10 seeds |
| `reference.py` | Value iteration with the true T and R. **Used only for grading**, never by the agent |
| `test_qlearning.py` | 8 fast unit tests (update rule, environment, matches Lab 3's AIMA solver) |

## How to run

**Requirements:** Python 3.10+ with **numpy** (the bundled `aima/mdp.py` imports it). Nothing else
is needed. On our machine that is the Anaconda Python, `/opt/anaconda3/bin/python3.13`. Any Python with
numpy works (`pip install numpy`).

```bash
git clone https://github.com/wilvalen/CS440Lab5.git
cd CS440Lab5

# 1. Train with the selected hyperparameters, print results, and animate one run (~1 s + animation)
python3 run_QLearning.py

# 2. Same thing, without the animation
python3 run_QLearning.py --no-animate

# 3. Try other hyperparameters (all flags are optional)
python3 run_QLearning.py --living-reward -2 --gamma 0.9 --alpha 0.1 --epsilon 0.2 \
                         --episodes 3000 --slip 0.2 --seed 0 --no-animate

# 4. Re-create the hyperparameter evidence (~20 s; --quick for ~3 s)
python3 hyperparameter_sweep.py

# 5. Unit tests (8 tests, < 1 s)
python3 -m unittest test_qlearning -v

# Lab 3 baseline (value iteration with known T and R)
python3 run_MDP.py
```

**What the output shows:** the learning curve (reward and goal rate per block of episodes), the learned
Q-values for each cell (Up/Down/Left/Right), the learned policy as arrows, the true optimal policy
for comparison, and summary numbers (% of states with the optimal action, V(start), goal rate).

**Claude Code features:** open the repo with `claude` (the CLI or the VS Code extension). The hook,
skill and sub-agent load automatically from `.claude/`. Then:

* edit any `.py` file: the hook runs the tests (it looks for a Python with numpy; set `LAB_PYTHON=/path/to/python` if it can't find one)
* type `/qlearning-experiment`, `/qlearning-experiment --epsilon 0.05`, or `/qlearning-experiment sweep`
* ask *"use the rl-reviewer agent to review the Q-learning code"*

## Claude Code features used (3 of 4: hooks, skills, sub-agents)

| Feature | Location | What it does |
|---|---|---|
| **Hook** | `.claude/settings.json`, `.claude/hooks/check_python.sh` | `PostToolUse` on `Edit\|Write\|MultiEdit`: after any `.py` edit it byte-compiles the file and runs the test suite. On failure it exits 2, so Claude sees the error and must fix it. (Tested: it catches a syntax error and a wrong-state bug in the update `max`.) |
| **Skill** | `.claude/skills/qlearning-experiment/SKILL.md` | `/qlearning-experiment [flags \| sweep]`: trains or sweeps with the right interpreter and reports the policy, metrics and an explanation in a fixed format. |
| **Sub-agent** | `.claude/agents/rl-reviewer.md` | `rl-reviewer`: a read-only reviewer that goes through a 7-item checklist. It confirms the agent never reads T or R, the TD target and terminal handling, ε-greedy, slip dynamics, and that the grader agrees with AIMA. It runs the tests and reports PASS or ISSUES FOUND. |
