import numpy as np


class Counter(object):

    def __init__(self,
                 env):
        self.env = env
        self.count_map = np.zeros((self.env.w, self.env.h))

    def increment(self, obs):
        idx = tuple(obs)
        self.count_map[idx] += 1

    def get_count(self, obs):
        idx = tuple(obs)
        return self.count_map[idx]

    def get_density(self, obs):
        count = self.get_count(obs)
        return count / sum(self.count_map)

    def get_density_map(self):
        if not (self.count_map > 0).any():
            return None
        return self.count_map / np.sum(self.count_map)
