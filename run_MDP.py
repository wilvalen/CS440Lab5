from environment import environment
from MDPAgent import MDPAgent;
from aima.utils import print_table
from print_details import draw_environment
from print_details import draw_policy
import time

if __name__ == "__main__":

    living_costs = [-0.05, -2, 2]
    # living_costs = [-0.05]
    for cost in living_costs:
        time.sleep(4)
        env = environment()
        env.set_MDP_env()
        print("Cost:", cost)
        V = MDPAgent(env, cost)
        draw_policy(V)

        env.insert_vacuum(V, 0,3)

        for i in range(15): #arbitrary max number of steps to not have it run forever
            action = V.choose_move(env)
            draw_environment(env, V)
            time.sleep(1)
            print("_______________")
            if action is None:
                print("reached end")
                print()
                break
            
            env.move(V,action)