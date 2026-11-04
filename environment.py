import random
import vacuum
from MDPAgent import MDPAgent;
from aima.utils import print_table
import time

# If the robot "slips", it moves in one of the two directions perpendicular to the one it chose.
PERPENDICULAR = {'Up': ('Left', 'Right'), 'Down': ('Left', 'Right'),
                 'Left': ('Up', 'Down'), 'Right': ('Up', 'Down')}

class environment():
    #creates environment
    def __init__(self):
        self.env = [[0,0,0,0],
               [0,0,0,0],
               [0,0,0,0],
               [0,0,0,0]]
    
    def set_MDP_env(self):
        self.env = [[0,0,0,0],
               [0,'X',0,0],
               [0,0,0,0],
               [0,-1,0,10]]

    # ---------------- Lab 4: reinforcement-learning environment ----------------
    # The agent is NOT given T or R. Both live only inside the environment, and the
    # agent can only find out about them by calling step() and seeing what happens.
    def set_RL_env(self, living_reward, slip=0.2, seed=None):
        """Lab 4 world: same layout as the MDP lab ('start' is the top-left cell).

        living_reward: reward for every step that does not end the episode.
        slip: probability the action goes sideways (slip/2 each way), as in AIMA's
              0.8 / 0.1 / 0.1 model. slip=0 makes the world deterministic.
        """
        self.set_MDP_env()
        self.living_reward = living_reward
        self.slip = slip
        self.rng = random.Random(seed)

    def is_terminal(self, row, col):
        cell = self.env[row][col]
        return cell != 'X' and cell != 0

    def step(self, V, action):
        """Apply action to vacuum V and return (reward, done).

        Bumping into a wall or the X leaves V where it is, and it still gets the living reward.
        Entering a terminal ends the episode and pays that cell's value (+10 or -1).
        """
        if self.slip and self.rng.random() < self.slip:
            action = self.rng.choice(PERPENDICULAR[action])
        self.move(V, action)
        if self.is_terminal(V.row, V.col):
            return self.env[V.row][V.col], True
        return self.living_reward, False

    # inserts vacuum
    def insert_vacuum(self, V, row, col):
        V.dirt_level = self.env[row][col]
        V.row = row
        V.col = col
        self.vacuum = V

    # shows perceptible squares from where the vacuum is
    def get_valid_states(self, V):
        row = V.row
        col = V.col

        curr = self.env[row][col]
        states = {}
        #check if valid move, if not, it isn't added. 
        if row > 0:
            if self.env[row - 1][col] != 'X':
                states['Up'] = self.env[row - 1][col]
        if row < 3:
            if self.env[row + 1][col] != 'X':
                states['Down'] = self.env[row + 1][col]
        if col > 0:
            if self.env[row][col - 1] != 'X':
                states['Left'] = self.env[row][col - 1]
        if col < 3:
            if self.env[row][col + 1] != 'X':
                states['Right'] = self.env[row][col + 1]
        
        return curr, states

    # actually moves the vacuum 
    def move(self, V, direction):
        _, states = self.get_valid_states(V)
        if direction not in states:
            return
        if direction == 'Up':
            V.row -= 1
        if direction == 'Down':
            V.row += 1
        if direction == 'Left':
            V.col -= 1
        if direction == 'Right':
            V.col += 1
        else:
            pass