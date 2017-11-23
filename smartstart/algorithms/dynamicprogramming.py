"""Value Iteration module

Describes ValueIteration class.

See 'Reinforcement Learning: An Introduction by Richard S. Sutton and
Andrew G. Barto for more information.
"""
import random
from collections import defaultdict

import numpy as np


class ValueIteration(object):
    """Value Iteration method
    
    Value iteration is a dynamic programming method. Requires full knowledge
    of the environment, i.e. transition model and reward function.

    Note:
        This implementation only works with one goal (terminal) state

    Parameters
    ----------
    env : :obj:`~smartstart.environments.environment.Environment`
        environment
    gamma : :obj:`float`
        discount factor
    min_error : :obj:`float`
        minimum error for convergence of value iteration
    max_itr : :obj:`int`
        maximum number of iteration of value iteration

    Attributes
    ----------
    env : :obj:`~smartstart.environments.environment.Environment`
        environment
    gamma : :obj:`float`
        discount factor
    min_error : :obj:`float`
        minimum error for convergence of value iteration
    max_itr : :obj:`int`
        maximum number of iteration of value iteration
    V : :obj:`collections.defaultdict`
        value function
    T : :obj:`collections.defaultdict`
        transition model
    R : :obj:`collections.defaultdict`
        reward function
    obses : :obj:`set`
        visited states
    goal : :obj:`tuple`
        goal state (terminal state)
    """

    def __init__(self,
                 env,
                 gamma=0.99,
                 min_error=1e-5,
                 max_itr=1000):
        self.env = env
        self.gamma = gamma
        self.min_error = min_error
        self.max_itr = max_itr

        self.V = defaultdict(lambda: 0)
        self.T = defaultdict(lambda: defaultdict(lambda: 0))
        self.R = defaultdict(lambda: 0)
        self.obses = set()
        self.goal = None

    def reset(self):
        """Reset internal state
        
        The following attributes are cleared:
        - self.V: Value function
        - self.T: Transition model
        - self.R: Reward function
        - self.obses: Visited observation
        - self.goal: Goal state

        """
        self.V.clear()
        self.T.clear()
        self.R.clear()
        self.goal = None

    def set_goal(self, obs):
        """Set goal state
        
        Goal state is added to the obses list and value function for the goal
        state is set to zero.

        Parameters
        ----------
        obs : :obj:`list` of :obj:`int` or :obj:`np.ndarray`
            observation

        """
        self.goal = tuple(obs)
        self.V[self.goal] = 0

    def set_transition(self, obs, action, obs_tp1, value):
        """Set transition of obs-action-obs_tp1
        
        Note:
            All transitions for a obs-action pair should add up to 1. This
            is not checked!

        Parameters
        ----------
        obs : :obj:`list` of :obj:`int` or :obj:`np.ndarray`
            observation
        action : :obj:`int`
            action
        obs_tp1 : :obj:`list` of :obj:`int` or :obj:`np.ndarray`
            next observation
        value : :obj:`float`
            transition probability

        """
        self.T[tuple(obs) + (action,)][tuple(obs_tp1)] = value

    def get_transition(self, obs, action, obs_tp1=None):
        """Returns transition probability of obs-action-obs_tp1

        Parameters
        ----------
        obs : :obj:`list` of :obj:`int` or :obj:`np.ndarray`
            observation
        action : :obj:`int`
            action
        obs_tp1 : :obj:`list` of :obj:`int` or :obj:`np.ndarray`
            next observation (Default value = None)

        Returns
        -------
        :obj:`float`
            Transition probability

        """
        if obs_tp1 is not None:
            return self.T[tuple(obs) + (action,)][tuple(obs_tp1)]

        return self.T[tuple(obs) + (action,)]

    def set_reward(self, obs, action, value):
        """Set reward for obs-action-obs_tp1

        Parameters
        ----------
        obs : :obj:`list` of :obj:`int` or :obj:`np.ndarray`
            observation
        action : :obj:`int`
            action
        obs_tp1 : :obj:`list` of :obj:`int` or :obj:`np.ndarray`
            next observation
        value : :obj:`float`
            reward

        """
        self.R[tuple(obs) + (action,)] = value

    def get_reward(self, obs, action):
        return self.R[tuple(obs) + (action,)]

    def optimize(self):
        """Run value iteration method
        
        Runs value iteration until converged or maximum number of iterations is
        reached. Method iterates through all visited states (self.obses).

        """
        for itr in range(self.max_itr):
            delta = 0
            for obs in self.obses:
                if obs != self.goal:
                    v = self.V[obs]
                    v_new = []
                    for action in self.env.possible_actions(obs):
                        v_new.append(self._get_value(obs, action))
                    self.V[obs] = max(v_new)
                    delta = max(delta, abs(v - self.V[obs]))
            if delta < self.min_error:
                # print("Converged in %d iterations" % itr)
                return
        # print("Did not converge in %d iterations" % self.max_itr)

    def get_action(self, obs):
        """Returns the policy for a certain observation
        
        Chooses the action that has the highest value function. When multiple
        actions have the same value function a random action is chosen from
        them.

        Parameters
        ----------
        obs : :obj:`list` of :obj:`int` or :obj:`np.ndarray`
            observation

        Returns
        -------
        :obj:`int`
            action

        """
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
        """Returns value for obs-action pair

        Parameters
        ----------
        obs : :obj:`list` of :obj:`int` or :obj:`np.ndarray`
            observation
        action : :obj:`int`
            action

        Returns
        -------
        :obj:`float`
            Value function for obs-action pair

        """
        if tuple(obs) == self.goal:
            return self.V[self.goal]
        value = 0.
        r = self.R[tuple(obs) + (action,)]
        for obs_prime, transition in self.get_transition(obs, action).items():
            value += transition * (r + self.gamma * self.V[obs_prime])
        return value

    def get_value_map(self):
        """Returns value map for environment
        
        Value-map will be a numpy array equal to the width (w) and height (h) of the environment.
        Each entry (state) will hold the value function associated with that
        state.

        Returns
        -------
        :obj:`numpy.ndarray`
            value map

        """
        V = np.zeros((self.env.w, self.env.h))
        for i in range(self.env.w):
            for j in range(self.env.h):
                V[i, j] = self.V[(i, j)]
        return V
