"""SARSA module

Module defining classes for SARSA and SARSA(lambda).

See 'Reinforcement Learning: An Introduction by Richard S. Sutton and
Andrew G. Barto for more information.
"""
from abc import ABCMeta

import numpy as np

from smartstart.algorithms.tdlearning import TDLearning


class TDTabular(TDLearning, metaclass=ABCMeta):

    def __init__(self,
                 env,
                 init_q_value=0.,
                 *args,
                 **kwargs):
        super(TDTabular, self).__init__(env, *args, **kwargs)
        self.init_q_value = init_q_value
        self.Q = np.ones(
            (self.env.w, self.env.h, self.env.num_actions)) * self.init_q_value

    def reset(self):
        """Resets Q-function

        The Q-function is set to the initial q-value for very state-action pair.
        """
        self.Q = np.ones(
            (self.env.w, self.env.h, self.env.num_actions)) * self.init_q_value

    def get_q_value(self, obs, action=None):
        """Returns Q-values and actions for observation obs

        Parameters
        ----------
        obs : :obj:`list` of :obj:`int` or :obj:`np.ndarray`
            observation

        Returns
        -------
        :obj:`list` of :obj:`float`
            Q-values
        :obj:`list` of :obj:`int`
            actions associated with each q-value in q_values

        """
        if action is not None:
            idx = tuple(obs) + (action,)
            return self.Q[idx], action

        actions = self.env.possible_actions(obs)
        q_values = []
        for action in actions:
            idx = tuple(obs) + (action,)
            q_values.append(self.Q[idx])
        return q_values, actions

    def update_q_value(self, obs, action, reward, obs_tp1, done):
        """Update Q-value for obs-action pair

        Updates Q-value according to the Bellman equation.
        Parameters
        ----------
        obs : :obj:`list` of :obj:`int` or :obj:`np.ndarray`
            observation
        action : :obj:`int`
            action
        reward : :obj:`float`
            reward
        obs_tp1 : :obj:`list` of :obj:`int` or :obj:`np.ndarray`
            next observation
        done : :obj:`bool`
            True when obs_tp1 is terminal
        Returns
        -------
        :obj:`float`
            updated Q-value and next action
        """
        next_q_value, action_tp1 = self.get_next_q_action(obs_tp1, done)
        q_value, _ = self.get_q_value(obs, action)
        td_error = self.alpha * (
            reward + self.gamma * next_q_value - q_value)

        idx = tuple(obs) + (action,)
        self.Q[idx] += self.alpha * td_error

        return self.Q[idx], action_tp1

    def get_q_map(self):
        return self.Q.copy()


class SARSA(TDTabular):
    """

    Parameters
    ----------
    env : :obj:`~smartstart.algorithms.tdlearning.TDLearning`
        environment
    *args :
        see parent :class:`~smartstart.algorithms.tdlearning.TDLearning`
    **kwargs :
        see parent :class:`~smartstart.algorithms.tdlearning.TDLearning`
    """

    def __init__(self, *args, **kwargs):
        super(SARSA, self).__init__(*args, **kwargs)

    def get_next_q_action(self, obs_tp1, done):
        """On-policy action selection

        Parameters
        ----------
        obs_tp1 : :obj:`list` of :obj:`int` or :obj:`np.ndarray`
            Next observation
        done : :obj:`bool`
            Boolean is True for terminal state

        Returns
        -------
        :obj:`float`
            Q-value for obs_tp1
        :obj:`int`
            action_tp1

        """
        if not done:
            action_tp1 = self.get_action(obs_tp1)
            next_q_value, _ = self.get_q_value(obs_tp1, action_tp1)
        else:
            next_q_value = 0.
            action_tp1 = None

        return next_q_value, action_tp1


class QLearning(TDTabular):
    """

    Parameters
    ----------
    env : :obj:`~smartstart.algorithms.tdlearning.TDLearning`
        environment
    *args :
        see parent :class:`~smartstart.algorithms.tdlearning.TDLearning`
    **kwargs :
        see parent :class:`~smartstart.algorithms.tdlearning.TDLearning`
    """

    def __init__(self, *args, **kwargs):
        super(QLearning, self).__init__(*args, **kwargs)

    def get_next_q_action(self, obs_tp1, done):
        """Off-policy action selection

        Parameters
        ----------
        obs_tp1 : :obj:`list` of :obj:`int` or :obj:`np.ndarray`
            Next observation
        done : :obj:`bool`
            Boolean is True for terminal state

        Returns
        -------
        :obj:`float`
            Q-value for obs_tp1
        :obj:`int`
            action_tp1

        """
        if not done:
            next_q_values, _ = self.get_q_value(obs_tp1)
            next_q_value = max(next_q_values)
            action_tp1 = self.get_action(obs_tp1)
        else:
            next_q_value = 0.
            action_tp1 = None

        return next_q_value, action_tp1


class SARSALambda(TDLearningLambda):
    """

    Parameters
    ----------
    env : :obj:`~smartstart.algorithms.tdlearning.TDLearning`
        environment
    *args :
        see parent :class:`~smartstart.algorithms.tdlearning
        .TDLearningLambda`
    **kwargs :
        see parent :class:`~smartstart.algorithms.tdlearning
        .TDLearningLambda`
    """

    def __init__(self, env, init_q_value=0., *args, **kwargs):
        super(SARSALambda, self).__init__(env, *args, **kwargs)
        self.init_q_value = init_q_value
        self.Q = np.ones(
            (self.env.w, self.env.h, self.env.num_actions)) * self.init_q_value
        self.traces = np.zeros((self.env.w, self.env.h, self.env.num_actions))

    def reset(self):
        """Resets Q-function

        The Q-function is set to the initial q-value for very state-action pair.
        """
        self.Q = np.ones(
            (self.env.w, self.env.h, self.env.num_actions)) * self.init_q_value

    def get_q_value(self, obs, action=None):
        """Returns Q-values and actions for observation obs

        Parameters
        ----------
        obs : :obj:`list` of :obj:`int` or :obj:`np.ndarray`
            observation

        Returns
        -------
        :obj:`list` of :obj:`float`
            Q-values
        :obj:`list` of :obj:`int`
            actions associated with each q-value in q_values

        """
        if action is not None:
            idx = tuple(obs) + (action,)
            return self.Q[idx], action

        actions = self.env.possible_actions(obs)
        q_values = []
        for action in actions:
            idx = tuple(obs) + (action,)
            q_values.append(self.Q[idx])
        return q_values, actions

    def get_next_q_action(self, obs_tp1, done):
        """On-policy action selection

        Parameters
        ----------
        obs_tp1 : :obj:`list` of :obj:`int` or :obj:`np.ndarray`
            Next observation
        done : :obj:`bool`
            Boolean is True for terminal state

        Returns
        -------
        :obj:`float`
            Q-value for obs_tp1
        :obj:`int`
            action_tp1

        """
        if not done:
            action_tp1 = self.get_action(obs_tp1)
            next_q_value, _ = self.get_q_value(obs_tp1, action_tp1)
        else:
            next_q_value = 0.
            action_tp1 = None

        return next_q_value, action_tp1

    def update_q_value(self, obs, action, reward, obs_tp1, done):
        """Update Q-value for obs-action pair

        Updates Q-value according to the Bellman equation with eligibility
        traces included.

        Parameters
        ----------
        obs : :obj:`list` of :obj:`int` or `np.ndarray`
            observation
        action : :obj:`int`
            action
        reward : :obj:`float`
            reward
        obs_tp1 : :obj:`list` of :obj:`int` or `np.ndarray`
            next observation
        done : :obj:`bool`
            True when obs_tp1 is terminal

        Returns
        -------
        :obj:`float`
            updated Q-value and next action

        """
        cur_q_value, _ = self.get_q_value(obs, action)
        next_q_value, action_tp1 = self.get_next_q_action(obs_tp1, done)
        td_error = reward + self.gamma * next_q_value - cur_q_value

        idx = tuple(obs) + (action,)
        self.traces[idx] = 1
        active_traces = np.asarray(
            np.where(self.traces > self.threshold_traces))
        for i in range(active_traces.shape[1]):
            idx = tuple(active_traces[:, i])
            self.Q[idx] += self.alpha * td_error * self.traces[idx]
            self.traces[idx] *= self.gamma * self.lamb

        return self.Q[idx], action_tp1

    def get_q_map(self):
        return self.Q.copy()
