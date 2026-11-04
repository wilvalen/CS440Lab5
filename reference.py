"""Ground truth for GRADING the Q-learner. The agent never uses this file.

Value iteration with the true T and R gives Q*(s,a), using the same definitions as
environment.step() (reward is paid when you enter a cell; terminals end the episode).
With these definitions, Q*(s,a) = sum_s' T(s'|s,a) * U(s'), where U is the utility from
AIMA's value_iteration in Lab 3. So the learned Q-table can be compared to it directly.
"""
from environment import PERPENDICULAR
from QLearningAgent import ACTIONS

MOVES = {'Up': (-1, 0), 'Down': (1, 0), 'Left': (0, -1), 'Right': (0, 1)}


def _go(grid, s, a):
    r, c = s[0] + MOVES[a][0], s[1] + MOVES[a][1]
    if 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] != 'X':
        return (r, c)
    return s


def transitions(grid, s, a, slip):
    outs = [(1 - slip, _go(grid, s, a))]
    if slip:
        outs += [(slip / 2, _go(grid, s, p)) for p in PERPENDICULAR[a]]
    return outs


def optimal_q(env, living_reward, gamma, slip, tol=1e-9):
    grid = env.env
    is_term = lambda s: grid[s[0]][s[1]] not in ('X', 0)
    states = [(r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] != 'X']
    reward = lambda s: grid[s[0]][s[1]] if is_term(s) else living_reward
    V = {s: 0.0 for s in states}
    while True:
        Q = {(s, a): sum(p * (reward(s2) + (0 if is_term(s2) else gamma * V[s2]))
                         for p, s2 in transitions(grid, s, a, slip))
             for s in states if not is_term(s) for a in ACTIONS}
        newV = {s: (0.0 if is_term(s) else max(Q[(s, a)] for a in ACTIONS)) for s in states}
        if max(abs(newV[s] - V[s]) for s in states) < tol:
            return Q
        V = newV


def policy_accuracy(agent, env, q_star, tol=1e-3):
    """Fraction of states where the learned greedy action is optimal (within tol, to allow ties)."""
    pi = agent.policy(env)
    good = sum(q_star[(s, a)] >= max(q_star[(s, b)] for b in ACTIONS) - tol for s, a in pi.items())
    return good / len(pi)


def evaluate_greedy(agent, env, episodes=500, start=(0, 0), max_steps=200):
    """Run the greedy policy with learning turned off. Returns (success rate, pit rate, mean return)."""
    results = [agent.run_episode(env, start, max_steps, learn=False) for _ in range(episodes)]
    goal = sum(r.outcome == 10 for r in results) / episodes
    pit = sum(r.outcome == -1 for r in results) / episodes
    mean_ret = sum(r.total_reward for r in results) / episodes
    return goal, pit, mean_ret
