"""Lab 4 driver: train the Q-learning vacuum, show what it learned, then run it.

    python3 run_QLearning.py                 # train with the selected values and animate one run
    python3 run_QLearning.py --no-animate    # print results only
    python3 run_QLearning.py --alpha 0.3 --epsilon 0.2 --living-reward -0.5 --gamma 0.95
"""
import argparse
import time

from environment import environment
from QLearningAgent import QLearningAgent, ACTIONS
from reference import optimal_q, policy_accuracy, evaluate_greedy
from print_details import draw_environment, draw_q_policy, draw_q_values

# ---------------------------------------------------------------- selected hyperparameters
# The reasons for each value are in README.md; hyperparameter_sweep.py shows the data behind them.
SELECTED = {
    'living_reward': -0.1,  # small cost per step: hurry up, but don't risk the -1 to save a step
    'gamma': 0.9,           # same as Lab 3; +10 found 6 steps away is still worth ~5.3 at the start
    'alpha': 0.1,           # moves are random (slip), so average many samples instead of chasing the newest
    'epsilon': 0.2,         # explore 20%: policy is optimal in 92% of states (86% at 0.1); above 0.2 gains are small
}
EPISODES = 3000
SLIP = 0.2          # hidden from the agent: 80% intended move, 10% each sideways
START = (0, 0)      # the 'start' cell, top-left


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--living-reward', type=float, default=SELECTED['living_reward'])
    ap.add_argument('--gamma', type=float, default=SELECTED['gamma'])
    ap.add_argument('--alpha', type=float, default=SELECTED['alpha'])
    ap.add_argument('--epsilon', type=float, default=SELECTED['epsilon'])
    ap.add_argument('--episodes', type=int, default=EPISODES)
    ap.add_argument('--slip', type=float, default=SLIP)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-animate', action='store_true')
    args = ap.parse_args()

    env = environment()
    env.set_RL_env(args.living_reward, slip=args.slip, seed=args.seed)
    agent = QLearningAgent(env, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon, seed=args.seed)

    print(f"living reward={args.living_reward}  gamma={args.gamma}  alpha={args.alpha}  "
          f"epsilon={args.epsilon}  episodes={args.episodes}  (hidden slip={args.slip})")
    history = agent.train(env, args.episodes, start=START)

    chunk = max(1, args.episodes // 10)
    print("\nLearning curve (mean reward per episode, and how often it reached +10):")
    for i in range(0, args.episodes, chunk):
        block = history[i:i + chunk]
        goal = sum(r.outcome == 10 for r in block) / len(block)
        print(f"  episodes {i + 1:>5}-{i + len(block):<5} reward {sum(r.total_reward for r in block) / len(block):6.2f}"
              f"   goal {goal:6.1%}")

    print("\nLearned Q-values  (Up / Down / Left / Right):")
    draw_q_values(env, agent)
    print("\nLearned greedy policy:")
    draw_q_policy(env, agent.policy(env))

    # Grading only: compare with the true optimum, which needs T and R (the agent never had them).
    q_star = optimal_q(env, args.living_reward, args.gamma, args.slip)
    opt_pi = {s: max(ACTIONS, key=lambda a: q_star[(s, a)]) for s in agent.policy(env)}
    print("\nOptimal policy from value iteration with the true T, R (for checking only):")
    draw_q_policy(env, opt_pi)
    goal, pit, ret = evaluate_greedy(agent, env, episodes=1000, start=START)
    v_star = max(q_star[(START, a)] for a in ACTIONS)
    print(f"\nLearned policy is optimal in {policy_accuracy(agent, env, q_star):.0%} of states")
    print(f"V(start): learned {agent.value(START):.3f}   true {v_star:.3f}")
    print(f"Greedy run, 1000 episodes: reached +10 {goal:.1%}, fell in -1 {pit:.1%}, mean reward {ret:.2f}")

    if not args.no_animate:
        print("\nOne greedy run from start:")
        env.insert_vacuum(agent, *START)
        for _ in range(30):
            draw_environment(env, agent)
            print("_______________")
            time.sleep(0.6)
            reward, done = env.step(agent, agent.choose_move(env, explore=False))
            if done:
                draw_environment(env, agent)
                print("reached end, reward", reward)
                break


if __name__ == "__main__":
    main()
