from aima.utils import print_table

def draw_environment(env, V):
    for row in range(4):
        for col in range(4):
            if row == V.row and col == V.col:
                print("\033[31mV\033[0m", end=" ")
            else:
                print(env.env[row][col], end=" ")
        print()

def draw_policy(V):
    print_table(V.mdp.to_arrows(V.policy))

# ---------------- Lab 4 (Q-learning) ----------------
ARROWS = {'Up': '^', 'Down': 'v', 'Left': '<', 'Right': '>'}

def _cell_label(env, r, c):
    cell = env.env[r][c]
    return 'X' if cell == 'X' else f"{cell:+d}" if env.is_terminal(r, c) else None

def draw_q_policy(env, policy):
    """policy: {(row, col): action}. Shows terminals as +10 / -1 and the obstacle as X."""
    table = [[_cell_label(env, r, c) or ARROWS[policy[(r, c)]] for c in range(4)] for r in range(4)]
    print_table(table)

def draw_q_values(env, agent):
    from QLearningAgent import ACTIONS
    table = [[_cell_label(env, r, c) or "/".join(f"{agent.q((r, c), a):.2f}" for a in ACTIONS)
              for c in range(4)] for r in range(4)]
    print_table(table)
