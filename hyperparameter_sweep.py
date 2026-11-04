"""Hyperparameter sweep for the Lab 4 Q-learner. It produces the evidence for the values chosen in run_QLearning.py.

Part 1 holds the selected values fixed and changes one hyperparameter at a time.
Part 2 changes the living reward, which changes the optimal policy itself (as in Lab 3).

    python3 hyperparameter_sweep.py            # full sweep (~30 s)
    python3 hyperparameter_sweep.py --quick    # fewer seeds/episodes
"""
import argparse
import statistics

from environment import environment
from QLearningAgent import QLearningAgent
from reference import optimal_q, policy_accuracy, evaluate_greedy
from run_QLearning import SELECTED, SLIP, START, EPISODES


def trial(living_reward, gamma, alpha, epsilon, episodes, seed):
    env = environment()
    env.set_RL_env(living_reward, slip=SLIP, seed=seed)
    agent = QLearningAgent(env, alpha=alpha, gamma=gamma, epsilon=epsilon, seed=seed)
    agent.train(env, episodes, start=START)
    q_star = optimal_q(env, living_reward, gamma, SLIP)
    acc = policy_accuracy(agent, env, q_star)
    goal, pit, ret = evaluate_greedy(agent, env, episodes=300, start=START)
    v_err = abs(agent.value(START) - max(q_star[(START, a)] for a in ('Up', 'Down', 'Left', 'Right')))
    return acc, goal, ret, v_err


def row(label, params, episodes, seeds):
    runs = [trial(**params, episodes=episodes, seed=s) for s in range(seeds)]
    acc, goal, ret, err = (statistics.mean(x) for x in zip(*runs))
    print(f"  {label:<22} policy-optimal {acc:6.1%}   goal-rate {goal:6.1%}   "
          f"mean return {ret:6.2f}   |V(start)-V*| {err:5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    seeds, episodes = (3, 1500) if args.quick else (10, EPISODES)

    print(f"Selected: {SELECTED}  slip={SLIP}  episodes={episodes}  seeds={seeds}\n")
    sweeps = {
        'alpha':   [0.01, 0.05, 0.1, 0.3, 0.5, 0.9],
        'epsilon': [0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
        'gamma':   [0.5, 0.7, 0.9, 0.99],
    }
    for name, values in sweeps.items():
        print(f"Varying {name} (others at selected values):")
        for v in values:
            row(f"{name}={v}" + ("  <- selected" if v == SELECTED[name] else ""),
                {**SELECTED, name: v}, episodes, seeds)
        print()

    print("Varying living_reward (this changes the optimal policy itself):")
    for lr in [0.5, 0.0, -0.04, -0.1, -0.5, -1.0, -2.0]:
        row(f"living={lr}" + ("  <- selected" if lr == SELECTED['living_reward'] else ""),
            {**SELECTED, 'living_reward': lr}, episodes, seeds)


if __name__ == '__main__':
    main()
