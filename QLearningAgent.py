"""Lab 4: Q-learning vacuum robot.

The agent does not know T (transition model) or R (reward function). It only sees
(state, action, reward, next_state, done) from environment.step() and uses each one
to update a Q-table:

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

If s' is terminal, the target is just r, because nothing comes after a terminal.
"""
import random
from collections import namedtuple

import vacuum

ACTIONS = ('Up', 'Down', 'Left', 'Right')

EpisodeResult = namedtuple('EpisodeResult', 'total_reward steps outcome')
# outcome: terminal reward reached (10 or -1), or None if the step limit hit first


class QLearningAgent(vacuum.vacuum):
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1, seed=None):
        super().__init__(env)
        self.alpha = alpha        # learning rate
        self.gamma = gamma        # discount factor
        self.epsilon = epsilon    # exploration rate for epsilon-greedy
        self.rng = random.Random(seed)
        self.Q = {}               # (state, action) -> value; unseen pairs count as 0

    # ---------------------------------------------------------------- Q-table helpers
    def state(self):
        return (self.row, self.col)

    def q(self, s, a):
        return self.Q.get((s, a), 0.0)

    def value(self, s):
        return max(self.q(s, a) for a in ACTIONS)

    def best_action(self, s):
        values = [self.q(s, a) for a in ACTIONS]
        best = max(values)
        # break ties randomly so the agent doesn't always pick 'Up' at the start
        return self.rng.choice([a for a, v in zip(ACTIONS, values) if v == best])

    # ---------------------------------------------------------------- acting / learning
    def choose_move(self, env, explore=True):
        """Epsilon-greedy: random action with prob. epsilon, else the greedy one."""
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(ACTIONS)
        return self.best_action(self.state())

    def update(self, s, a, reward, s_next, done):
        target = reward if done else reward + self.gamma * self.value(s_next)
        old = self.q(s, a)
        self.Q[(s, a)] = old + self.alpha * (target - old)

    def run_episode(self, env, start=(0, 0), max_steps=200, learn=True):
        """Run one episode from start. learn=False means act greedily and don't update Q."""
        env.insert_vacuum(self, *start)
        total, outcome = 0.0, None
        for step in range(1, max_steps + 1):
            s = self.state()
            a = self.choose_move(env, explore=learn)
            reward, done = env.step(self, a)
            if learn:
                self.update(s, a, reward, self.state(), done)
            total += reward
            if done:
                outcome = reward
                break
        return EpisodeResult(total, step, outcome)

    def train(self, env, episodes, start=(0, 0), max_steps=200):
        return [self.run_episode(env, start, max_steps, learn=True) for _ in range(episodes)]

    def policy(self, env):
        """Greedy action for every non-terminal, non-X cell. The env is used only to lay out the grid."""
        pi = {}
        for r in range(len(env.env)):
            for c in range(len(env.env[0])):
                if env.env[r][c] == 'X' or env.is_terminal(r, c):
                    continue
                s = (r, c)
                values = [self.q(s, a) for a in ACTIONS]
                pi[s] = ACTIONS[values.index(max(values))]
        return pi
