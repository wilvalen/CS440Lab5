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