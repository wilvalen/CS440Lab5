"""Fast checks for Lab 4 (< 1 s). Run: python3 -m unittest test_qlearning -v"""
import unittest

from environment import environment
from QLearningAgent import QLearningAgent, ACTIONS
from reference import optimal_q, policy_accuracy, evaluate_greedy
from run_QLearning import SELECTED, START


def make(living=-0.1, slip=0.0, seed=0, **kw):
    env = environment()
    env.set_RL_env(living, slip=slip, seed=seed)
    return env, QLearningAgent(env, seed=seed, **kw)


class TestEnvironment(unittest.TestCase):
    def test_walls_and_obstacle_block_but_cost_living_reward(self):
        env, a = make()
        env.insert_vacuum(a, 0, 0)
        self.assertEqual(env.step(a, 'Up'), (-0.1, False))
        self.assertEqual(a.state(), (0, 0))
        env.insert_vacuum(a, 0, 1)
        env.step(a, 'Down')              # (1,1) is the X
        self.assertEqual(a.state(), (0, 1))

    def test_terminals_pay_and_end(self):
        env, a = make()
        env.insert_vacuum(a, 3, 2)
        self.assertEqual(env.step(a, 'Right'), (10, True))
        env.insert_vacuum(a, 3, 0)
        self.assertEqual(env.step(a, 'Right'), (-1, True))

    def test_slip_frequency(self):
        env, a = make(slip=0.2, seed=1)
        moved_right = 0
        for _ in range(5000):
            env.insert_vacuum(a, 2, 2)
            env.step(a, 'Up')
            moved_right += a.state() == (2, 3)
        self.assertAlmostEqual(moved_right / 5000, 0.1, delta=0.02)


class TestAgent(unittest.TestCase):
    def test_update_rule(self):
        env, a = make(alpha=0.5, gamma=0.9)
        a.Q[((0, 1), 'Right')] = 4.0
        a.update((0, 0), 'Right', -0.1, (0, 1), done=False)
        self.assertAlmostEqual(a.q((0, 0), 'Right'), 0.5 * (-0.1 + 0.9 * 4.0))
        a.update((3, 2), 'Right', 10, (3, 3), done=True)   # terminal: no bootstrap
        self.assertAlmostEqual(a.q((3, 2), 'Right'), 5.0)

    def test_epsilon_zero_is_greedy(self):
        env, a = make(epsilon=0.0)
        env.insert_vacuum(a, 0, 0)
        a.Q[((0, 0), 'Down')] = 1.0
        self.assertTrue(all(a.choose_move(env) == 'Down' for _ in range(50)))

    def test_deterministic_world_learns_exact_optimum(self):
        env, a = make(slip=0.0, alpha=0.5, gamma=0.9, epsilon=0.3)
        a.train(env, 1500, start=START)
        q_star = optimal_q(env, -0.1, 0.9, 0.0)
        self.assertEqual(policy_accuracy(a, env, q_star), 1.0)
        self.assertAlmostEqual(a.value(START), max(q_star[(START, x)] for x in ACTIONS), places=2)

    def test_selected_hyperparameters_solve_lab(self):
        env, a = make(living=SELECTED['living_reward'], slip=0.2,
                      **{k: SELECTED[k] for k in ('alpha', 'gamma', 'epsilon')})
        a.train(env, 3000, start=START)
        goal, pit, _ = evaluate_greedy(a, env, episodes=300, start=START)
        self.assertGreater(goal, 0.95)
        self.assertLess(pit, 0.05)


class TestReference(unittest.TestCase):
    def test_matches_aima_value_iteration(self):
        """The grader's Q* agrees with Lab 3's AIMA GridMDP: Q*(s,a) = sum T(s'|s,a) U(s')."""
        from MDPAgent import MDPAgent
        env, _ = make(living=-0.1)
        mdp_agent = MDPAgent(env, -0.1, gamma=0.9)
        q_star = optimal_q(env, -0.1, 0.9, 0.2)
        for (r, c) in {s for s, _ in q_star}:
            aima = mdp_agent.U[(c, 3 - r)]
            mine = max(q_star[((r, c), x)] for x in ACTIONS)
            self.assertAlmostEqual(mine, (aima - (-0.1)) / 0.9, delta=0.05)


if __name__ == '__main__':
    unittest.main()
