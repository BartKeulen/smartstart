"""Q-Learning module

Module defining classes for Q-Learning and Q(lambda).

See 'Reinforcement Learning: An Introduction by Richard S. Sutton and
Andrew G. Barto for more information.
"""
import numpy as np

from smartstart.algorithms.tabular.sarsa import SARSA


class QLearning(SARSA):
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

    def __init__(self, env, init_q_value=0., *args, **kwargs):
        super(QLearning, self).__init__(env, init_q_value, *args, **kwargs)
        self.init_q_value = init_q_value
        self.Q = np.ones(
            (self.env.w, self.env.h, self.env.num_actions)) * self.init_q_value

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


# class QLearningLambda(TDLearningLambda):
#     """
#     Note:
#         Does not work properly, because q-learning is off-policy standard
#         eligibility traces might fail.
#
#     Parameters
#     ----------
#     env : :obj:`~smartstart.algorithms.tdlearning.TDLearning`
#         environment
#     *args :
#         see parent :class:`~smartstart.algorithms.tdlearning
#         .TDLearningLambda`
#     **kwargs :
#         see parent :class:`~smartstart.algorithms.tdlearning
#         .TDLearningLambda`
#     """
#
#     def __init__(self, env, *args, **kwargs):
#         super(QLearningLambda, self).__init__(env, *args, **kwargs)
#
#     def get_next_q_action(self, obs_tp1, done):
#         """Off-policy action selection
#
#         Parameters
#         ----------
#         obs_tp1 : :obj:`list` of :obj:`int` or :obj:`np.ndarray`
#             Next observation
#         done : :obj:`bool`
#             Boolean is True for terminal state
#
#         Returns
#         -------
#         :obj:`float`
#             Q-value for obs_tp1
#         :obj:`int`
#             action_tp1
#
#         """
#         if not done:
#             next_q_values, _ = self.get_q_value(obs_tp1)
#             next_q_value = max(next_q_values)
#             action_tp1 = self.get_action(obs_tp1)
#         else:
#             next_q_value = 0.
#             action_tp1 = None
#
#         return next_q_value, action_tp1
