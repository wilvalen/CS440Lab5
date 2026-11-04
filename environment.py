import numpy as np
import vacuum
from MDPAgent import MDPAgent;
from aima.utils import print_table
import time

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