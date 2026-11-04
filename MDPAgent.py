import vacuum
from aima.mdp import GridMDP, value_iteration, best_policy

SIZE = 4

class MDPAgent(vacuum.vacuum):
    def __init__(self, env, living_cost, gamma =0.9):
        super().__init__(env)
        self.living_cost = living_cost
        self.gamma = gamma
        grid, terminals = self.build_grid(env)
        self.mdp = GridMDP(grid,terminals, gamma=gamma)
        self.U = value_iteration(self.mdp)
        self.policy = best_policy(self.mdp, self.U)



    def choose_move(self, env):
        x = self.col
        y = SIZE - 1 - self.row
        best_move = self.policy[(x,y)]
        if best_move == (0, 1):
            return 'Up'
        elif best_move == (0,-1):
            return 'Down'
        elif best_move == (1,0):
            return 'Right'
        elif best_move == (-1,0):
            return 'Left'
        else:
            return None # case where there is no best move, has finished.
        

        return

    def build_grid(self,env):
        grid = []
        terminals = []
        for row in range(SIZE):
            new_row = []
            for col in range(SIZE):
                cell = env.env[row][col]
                if cell == 'X':
                    new_row.append(None)
                elif cell == 0:
                    new_row.append(self.living_cost)
                else:
                    new_row.append(cell)
                    terminals.append((col, SIZE-1-row))
            grid.append(new_row)
        return grid, terminals