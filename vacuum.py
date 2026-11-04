
class vacuum:
    def __init__(self,enviornment):
        self.dirt_level = 0
        self.row = 0
        self.col = 0

    def choose_move(self,current,neighbors):
        #should be overwritten by hillclimb and annealing. Nothing else should need to be added here
        raise NotImplementedError
    
