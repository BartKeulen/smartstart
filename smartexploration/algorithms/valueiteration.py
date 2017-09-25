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
    """

    def __init__(self,
                 env,
                 gamma=0.99,
                 min_error=1e-5,
                 max_itr=1000):
        """Constructs ValueIteration object

        Args:
            env:        environment
            gamma:      discount factor
            min_error:  minimum error for convergence of value iteration
            max_itr:    maximum number of iteration of value iteration
        """
        self.env = env
        self.gamma = gamma
        self.min_error = min_error
        self.max_itr = max_itr

        self.V = defaultdict(lambda: 0)
        self.T = defaultdict(lambda: defaultdict(lambda: 0))
        self.R = defaultdict(lambda: defaultdict(lambda: 0))
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
        self.obses.clear()
        self.goal = None

    def add_obs(self, obs):
        """Adds observation to the obses set

        Args:
            obs:    observation
        Returns:

        """
        self.obses.add(tuple(obs))

    def set_goal(self, obs):
        """Set goal state

        Goal state is added to the obses list and value function for the goal
        state is set to zero.

        Args:
            obs:    observation

        """
        self.add_obs(obs)
        self.goal = tuple(obs)
        self.V[self.goal] = 0

    def set_transition(self, obs, action, obs_tp1, value):
        """Set transition of obs-action-obs_tp1

        Note:
            All transitions for a obs-action pair should add up to 1. This
            is not checked!

        Args:
            obs:        observation
            action:     action
            obs_tp1:    next observation
            value:      transition probability
        """
        self.T[tuple(obs) + (action,)][tuple(obs_tp1)] = value

    def get_transition(self, obs, action, obs_tp1=None):
        """Returns transition probability of obs-action-obs_tp1

        Args:
            obs:        observation
            action:     action
            obs_tp1:    next observation

        Returns:
            Transition probability
        """
        if obs_tp1 is not None:
            return self.T[tuple(obs) + (action,)][tuple(obs_tp1)]

        return self.T[tuple(obs) + (action,)]

    def set_reward(self, obs, action, obs_tp1, value):
        """Set reward for obs-action-obs_tp1

        Args:
            obs:        observation
            action:     action
            obs_tp1:    next observation
            value:      reward
        """
        self.R[tuple(obs) + (action,)][tuple(obs_tp1)] = value

    def optimize(self):
        """Run value iteration method

        Runs value iteration until converged or maximum number of iterations is
        reached. Method iterates through all visited states (self.obses).
        """
        for _ in range(self.max_itr):
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
                return

    def get_action(self, obs):
        """Returns the policy for a certain observation

        Chooses the action that has the highest value function. When multiple
        actions have the same value function a random action is chosen from
        them.

        Args:
            obs:    observation

        Returns:
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

        Args:
            obs:        observation
            action:     action

        Returns:
            Value function for obs-action pair
        """
        if tuple(obs) == self.goal:
            return self.V[self.goal]
        value = 0.
        for obs_prime, transition in self.get_transition(obs, action).items():
            r = self.R[tuple(obs) + (action,)][tuple(obs_prime)]
            value += transition * (r + self.gamma * self.V[obs_prime])
        return value

    def get_value_map(self):
        """Returns value map for environment

        Value-map will be a numpy array equal to the width (w) and height (h) of the environment.
        Each entry (state) will hold the value function associated with that
        state.

        Returns:
            numpy.ndarray:  value map
        """
        V = np.zeros((self.env.w, self.env.h))
        for i in range(self.env.w):
            for j in range(self.env.h):
                V[i, j] = self.V[(i, j)]
        return V


# if __name__ == "__main__":
#     import time
#     from smartexploration.environments.gridworld import GridWorld, \
#         GridWorldVisualizer
#
#     env = GridWorld.generate(GridWorld.IMPOSSIBRUUHHH)
#     visualizer = GridWorldVisualizer(env)
#     visualizer.add_visualizer(GridWorldVisualizer.LIVE_AGENT,
#                               GridWorldVisualizer.VALUE_FUNCTION,
#                               GridWorldVisualizer.DENSITY,
#                               GridWorldVisualizer.CONSOLE)
#
#     env.visualizer = visualizer
#     # env.T_prob = 0.1
#     env.reset()
#
#     algo = ValueIteration(env, max_itr=10000, min_error=1e-5)
#     algo.T, algo.R = env.get_T_R()
#     algo.obses = env.get_all_states()
#     algo.set_goal(env.goal_state)
#
#     algo.optimize()
#
#     density_map = np.zeros((env.w, env.h))
#
#     obs = env.reset()
#     density_map[tuple(obs)] += 1
#     env.render(value_map=algo.get_value_map(), density_map=density_map)
#     steps = 0
#     while True:
#         obs, reward, done, _ = env.step(algo.get_action(obs))
#         steps += 1
#         density_map[tuple(obs)] += 1
#         env.render(value_map=algo.get_value_map(), density_map=density_map)
#
#         time.sleep(0.01)
#
#         if done:
#             break
#
#     print("Optimal path for %s is %d steps in length" % (env.name, steps))
#
#     render = True
#     while render:
#         render = env.render(value_map=algo.get_value_map(),
#                             density_map=density_map)
