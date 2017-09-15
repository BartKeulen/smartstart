import random
from collections import defaultdict

import numpy as np


class ValueIteration(object):

    def __init__(self,
                 env,
                 gamma=0.99,
                 min_error=1e-5,
                 max_itr=100):
        self.env = env
        self.gamma = gamma
        self.min_error = min_error
        self.max_itr = max_itr

        # self.V = np.zeros((self.env.w, self.env.h))
        self.V = defaultdict(lambda: 0)
        self.T = defaultdict(lambda: defaultdict(lambda: 0))
        self.R = defaultdict(lambda: 0)
        self.obses = set()

    def reset(self):
        self.V.clear()
        self.T.clear()
        self.R.clear()

    def add_obs(self, obs):
        self.obses.add(tuple(obs))

    def set_transition(self, obs, action, obs_tp1, value):
        self.T[tuple(obs) + (action,)][tuple(obs_tp1)] = value

    def get_transition(self, obs, action, obs_tp1=None):
        if obs_tp1 is not None:
            return self.T[tuple(obs) + (action,)][tuple(obs_tp1)]

        return self.T[tuple(obs) + (action,)]

    def set_reward(self, obs, action, value):
        self.R[tuple(obs) + (action,)] = value

    def optimize(self):
        for itr in range(self.max_itr):
            delta = 0
            for obs in self.obses:
                v = self.V[obs]
                v_new = []
                for action in self.env.possible_actions(obs):
                    v_new.append(self._get_value(obs, action))
                self.V[obs] = max(v_new)
                delta = max(delta, abs(v - self.V[obs]))
            if delta < self.min_error:
                print("DELTA:", delta)
                break

    def get_action(self, obs):
        actions = self.env.possible_actions(obs)
        max_actions = []
        max_value = -float('inf')
        for action in actions:
            value = self._get_value(obs, action)
            if value > max_value:
                max_value = value
                max_actions = [action]
            elif value == max_value:
                max_actions.append(action)

        return random.choice(max_actions)

    def _get_value(self, obs, action):
        r = self.R[tuple(obs) + (action,)]
        value = 0.
        for obs_prime, transition in self.get_transition(obs, action).items():
            value += transition * (r + self.gamma * self.V[obs_prime])
        return value

    def get_value_map(self):
        V = np.zeros((self.env.w, self.env.h))
        for i in range(self.env.w):
            for j in range(self.env.h):
                V[i, j] = self.V[(i, j)]
        return V