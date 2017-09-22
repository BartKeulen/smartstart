from collections import defaultdict

import numpy as np


class Counter(object):

    def __init__(self, env):
        self.env = env
        self.count_map = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        self.total = 0

    def increment(self, obs, action, obs_tp1):
        self.count_map[tuple(obs)][(action,)][tuple(obs_tp1)] += 1
        self.total += 1

    def get_count(self, obs, action=None, obs_tp1=None):
        if action is None:
            total = 0
            for value in self.count_map[tuple(obs)].values():
                total += sum(value.values())
            return total
        if obs_tp1 is None:
            return sum(self.count_map[tuple(obs)][(action,)].values())
        else:
            return self.count_map[tuple(obs)][(action,)][tuple(obs_tp1)]

    def get_density(self, obs, action=None):
        count = np.sum(self.get_count(obs, action))
        return count / self.total

    def get_count_map(self):
        count_map = np.zeros((self.env.w, self.env.h), dtype=np.int)
        for i in range(self.env.w):
            for j in range(self.env.h):
                count_map[i, j] = self.get_count([i, j])
        return count_map

    def get_density_map(self):
        count_map = self.get_count_map()
        if np.sum(count_map) == 0:
            return count_map
        return count_map / self.total