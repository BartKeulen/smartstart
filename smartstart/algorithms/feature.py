from abc import ABCMeta, abstractmethod

import numpy as np


class Feature(metaclass=ABCMeta):

    @abstractmethod
    def get(self, obs):
        raise NotImplementedError


class TriangularFeature(Feature):

    def __init__(self, num_states, obses_min, obses_max, num_features_dim):
        self.num_features_dim = num_features_dim
        self.features = np.zeros((num_states, self.num_features_dim))
        for i, (obs_min, obs_max) in enumerate(zip(obses_min, obses_max)):
            self.features[i, :] = np.linspace(obs_min, obs_max, self.num_features_dim)
        self.step_sizes = (np.asarray(obses_max) - np.asarray(obses_min)) / (num_features_dim - 1)

    def get(self, obs):
        feature_vector = np.zeros(self.features.shape)
        for i, single_obs in enumerate(obs):
            idx = np.searchsorted(self.features[i, :], single_obs)
            if idx >= self.num_features_dim:
                feature_vector[i, idx-1] = 1.
                continue
            remainder = self.features[i, idx] - single_obs
            if idx == 0 or remainder == 0:
                feature_vector[i, idx] = 1.
            else:
                feature_vector[i, idx] = 1 - remainder/self.step_sizes[i]
                feature_vector[i, idx - 1] = remainder/self.step_sizes[i]
        return np.ravel(feature_vector)
