"""Counter module

Describes Counter base class for TD-learning algorithms
"""
from collections import defaultdict

import numpy as np


class Counter(object):
    """Base class for visitation counts.

    Base class for keeping track of obs-action-obs_tp1 visitation counts in
    discrete state and discrete action reinforcement learning algorithms.
    """

    def __init__(self, env):
        """Constructs Counter object

        Args:
            env:        environment
        """
        self.env = env
        self.count_map = defaultdict(lambda:
                                     defaultdict(lambda:
                                                 defaultdict(lambda: 0)))
        self.total = 0

    def increment(self, obs, action, obs_tp1):
        """Increment count for obs-action-obs_tp1 transition.

        Args:
            obs:        current observation
            action:     current action
            obs_tp1:    next observation
        """
        self.count_map[tuple(obs)][(action,)][tuple(obs_tp1)] += 1
        self.total += 1

    def get_count(self, obs, action=None, obs_tp1=None):
        """Returns visitation count

        Visitation count for obs, obs-action or obs-action-obs_tp1 is returned.
        User can leave action and/or obs_tp1 empty to return a higher level
        count.

        Note:
            When action is None and obs_tp1 is not None, the count for just obs
            will be returned and obs_tp1 will not be taken into account.

        Args:
            obs:        observation
            action:     action
            obs_tp1:    next observation

        Returns:
            Visitation count

        """
        if action is None:
            total = 0
            for value in self.count_map[tuple(obs)].values():
                total += sum(value.values())
            return total
        if obs_tp1 is None:
            return sum(self.count_map[tuple(obs)][(action,)].values())
        else:
            return self.count_map[tuple(obs)][(action,)][tuple(obs_tp1)]

    def get_density(self, obs, action=None, obs_tp1=None):
        """Density for obs, obs-action or obs-action-obs_tp1 is returned.

        Density is calculated by dividing the count with the total count.

        Args:
            obs:        observation
            action:     action
            obs_tp1:    next observation

        Returns:
            Density
        """
        count = np.sum(self.get_count(obs, action, obs_tp1))
        return count / self.total

    def get_count_map(self):
        """Returns state count map for environment.

        Count-map will be a numpy array equal to the width (w) and height (h)
        of the environment. Each entry (state) will hold the count associated
        with that state.

        Note:
            Only works for 2D-environments that have a w and h attribute.

        Returns:
            Count map
        """
        count_map = np.zeros((self.env.w, self.env.h), dtype=np.int)
        for i in range(self.env.w):
            for j in range(self.env.h):
                count_map[i, j] = self.get_count([i, j])
        return count_map

    def get_density_map(self):
        """Returns state density map for environment

        Density-map will be a numpy array equal to the width (w) and height (h)
        of the environment. Each entry (state) will hold the density
        associated with that state.

        Note:
            Only works for 2D-environments that have a w and h attribute.

        Returns:
            Density map
        """
        count_map = self.get_count_map()
        if np.sum(count_map) == 0:
            return count_map
        return count_map / self.total
