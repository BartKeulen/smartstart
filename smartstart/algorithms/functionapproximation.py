from abc import ABCMeta, abstractmethod

import numpy as np

from smartstart.algorithms.tdlearning import TDLearning


class FunctionApproximation(TDLearning, metaclass=ABCMeta):

    def __init__(self,
                 env,
                 feature,
                 *args,
                 **kwargs):
        super(FunctionApproximation, self).__init__(env, *args, **kwargs)
        self.env = env
        self.feature = feature
        self.num_actions = env.num_actions
        self.theta = np.zeros((self.feature.size, env.num_actions))

    def reset(self):
        self.theta = np.zeros((self.feature.size, self.num_actions))

    def get_q_value(self, obs, action=None):
        feature_vector = self.feature.get(obs)
        if action is None:
            return self.theta.T.dot(feature_vector), np.asarray(range(self.theta.shape[1]))

        return self.theta[:, action].T.dot(feature_vector)

    def update_q_value(self, obs, action, reward, obs_tp1, done):
        next_q_value, action_tp1 = self.get_next_q_action(obs_tp1, done)
        td_error = reward + self.gamma * next_q_value - self.get_q_value(obs, action)

        self.theta[:, action] += self.alpha * td_error * self.feature.get(obs)
        return self.get_q_value(obs, action), action_tp1

    def take_step(self, obs, action, episode, render=False):
        obs_tp1, reward, done = self.env.step(action)

        if render:
            self.render()

        _, action_tp1 = self.update_q_value(obs, action, reward, obs_tp1, done)

        episode.add(reward)

        return obs_tp1, action_tp1, done, render

    def render(self, message=None):
        return self.env.render()

    def get_q_map(self):

        pass


class SARSAFA(FunctionApproximation):

    def __init__(self, env, *args, **kwargs):
        super(SARSAFA, self).__init__(env, *args, **kwargs)

    def get_next_q_action(self, obs_tp1, done):
        if not done:
            action_tp1 = self.get_action(obs_tp1)
            next_q_value = self.get_q_value(obs_tp1, action_tp1)
        else:
            next_q_value = 0.
            action_tp1 = None
        return next_q_value, action_tp1


class QLearningFA(FunctionApproximation):

    def __init__(self, env, *args, **kwargs):
        super(QLearningFA, self).__init__(env, *args, **kwargs)

    def get_next_q_action(self, obs_tp1, done):
        if not done:
            next_q_values, _ = self.get_q_value(obs_tp1)
            next_q_value = max(next_q_values)
            action_tp1 = self.get_action(obs_tp1)
        else:
            next_q_value = 0.
            action_tp1 = None

        return next_q_value, action_tp1
