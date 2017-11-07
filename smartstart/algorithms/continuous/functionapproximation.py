from abc import ABCMeta, abstractmethod

import numpy as np

from smartstart.algorithms.tdlearning import TDLearning


class FunctionApproximation(TDLearning, metaclass=ABCMeta):

    def __init__(self,
                 env,
                 num_actions,
                 feature,
                 alpha=0.1,
                 gamma=0.99):

        self.env = env
        self.feature = feature
        self.theta = np.zeros((self.feature.num_features, num_actions))
        self.alpha = alpha
        self.gamma = gamma

    def get_q_value(self, obs, action=None):
        feature_vector = self.feature.get(obs)
        if action is None:
            return self.theta.T.dot(feature_vector)

        return self.theta[:, action].T.dot(feature_vector)

    def update_q_value(self, obs, action, reward, obs_tp1, done):
        next_q_value, action_tp1 = self.get_next_q_action(obs_tp1, done)
        td_error = reward + self.gamma * next_q_value - self.get_q_value(obs, action)

        self.theta[:, action] += self.alpha * td_error * self.feature.get(obs)
        return self.get_q_value(obs, action), action_tp1

    @abstractmethod
    def get_next_q_action(self, obs_tp1, done):
        raise NotImplementedError
