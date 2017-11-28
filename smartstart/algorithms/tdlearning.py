"""Temporal-Difference module

Describes TDLearning and TDLearningLambda base classes for temporal
difference learning without and with eligibility traces.

See 'Reinforcement Learning: An Introduction by Richard S. Sutton and
Andrew G. Barto for more information.
"""
import random
from abc import ABCMeta, abstractmethod

import numpy as np

from smartstart.utilities.datacontainers import Summary, Episode
from smartstart.utilities.scheduler import Scheduler


class Base(object):

    def __init__(self, *args, **kwargs):
        pass


class TDLearning(Base, metaclass=ABCMeta):
    """Base class for temporal-difference methods
    
    Base class for temporal difference methods Q-Learning, SARSA,
    SARSA(lambda) and Q(lambda). Implements all common methods, specific
    methods to each algorithm have to be implemented in the child class.
    
    All the exploration methods are defined in this class and can be added by
    adding a new method that describes the exploration strategy. The
    exploration strategy must be added to the class attribute below and
    inserted in the get_action method.

    Currently 5 exploration methods are implemented; no exploration,
    epsilon-greedy, boltzmann, count-based and ucb. Another exploration
    method is optimism in the face of uncertainty which can be used by
    setting the init_q_value > 0.

    Parameters
    ----------
    env : :obj:`~smartstart.environments.environment.Environment`
        environment
    num_episodes : :obj:`int`
        number of episodes
    max_steps : :obj:`int`
        maximum number of steps per episode
    alpha : :obj:`float`
        learning step-size
    gamma : :obj:`float`
        discount factor
    init_q_value : :obj:`float`
        initial q-value
    exploration : :obj:`str`
        exploration strategy, see class attributes for available options
    epsilon : :obj:`float` or :obj:`Scheduler`
        epsilon-greedy parameter
    temp : :obj:`float`
        temperature parameter for Boltzmann exploration
    beta : :obj:`float`
        count-based exploration parameter

    Attributes
    -----------
    max_steps : :obj:`int`
        number of episodes
    steps_episode : :obj:`int`
        maximum number of steps per episode
    alpha : :obj:`float`
        learning step-size
    gamma : :obj:`float`
        discount factor
    init_q_value : :obj:`float`
        initial q-value
    Q : :obj:`np.ndarray`
        Numpy ndarray holding the q-values for all state-action pairs
    exploration : :obj:`str`
        exploration strategy, see class attributes for available options
    epsilon : :obj:`float` or :obj:`Scheduler`
        epsilon-greedy parameter
    temp : :obj:`float`
        temperature parameter for Boltzmann exploration
    beta : :obj:`float`
        count-based exploration parameter
    """
    NONE = "None"
    E_GREEDY = "E-Greedy"
    BOLTZMANN = "Boltzmann"

    def __init__(self,
                 env,
                 max_steps=500000,
                 steps_episode=1000,
                 alpha=0.1,
                 gamma=0.99,
                 exploration_strategy=E_GREEDY,
                 epsilon=0.1,
                 temp=0.5):
        super(TDLearning, self).__init__()
        self.env = env
        self.max_steps = max_steps
        self.steps_episode = steps_episode
        self.alpha = alpha
        self.gamma = gamma
        self.exploration_strategy = exploration_strategy
        self.epsilon = epsilon
        self.temp = temp

        self.test_render = False

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def get_q_value(self, obs, action=None):
        raise NotImplementedError

    @abstractmethod
    def get_next_q_action(self, obs_tp1, done):
        raise NotImplementedError

    @abstractmethod
    def update_q_value(self, obs, action, reward, obs_tp1, done):
        raise NotImplementedError

    def train(self, test_freq=0, render=False, render_episode=False, print_results=True):
        """Runs a training experiment
        
        Training experiment runs for self.num_episodes and each episode takes
        a maximum of self.max_steps.

        Parameters
        ----------
        render : :obj:`bool`
            True when rendering every time-step (Default value = False)
        render_episode : :obj:`bool`
            True when rendering every episode (Default value = False)
        print_results : :obj:`bool`
            True when printing results to console (Default value = True)

        Returns
        -------
        :class:`~smartexploration.utilities.datacontainers.Summary`
            Summary Object containing the training data

        """
        summary = Summary(self.__class__.__name__, self.env.name, self.exploration_strategy)

        i_episode = 0
        total_steps = 0
        while total_steps < self.max_steps:
            episode = Episode(i_episode)

            obs = self.env.reset()
            action = self.get_action(obs)

            for _ in range(self.steps_episode):
                obs, action, done, render = self.take_step(obs, action, episode,
                                                           render)

                total_steps += 1

                if done:
                    break

            # Add training episode to summary
            summary.append(episode)

            # Render and/or print results
            message = "Episode: %d, steps: %d, reward: %.2f" % (
                i_episode, len(episode), episode.reward)
            if render or render_episode:
                render_episode = self.render(message=message)

            if print_results:
                print(message)

            # Run test episode and add tot summary
            if test_freq != 0 and (i_episode % test_freq == 0 or total_steps >= self.max_steps):
                test_episode = self.run_test_episode(i_episode)
                summary.append_test(test_episode)

                if print_results:
                    print(
                        "TEST Episode: %d, steps: %d, reward: %.2f" % (
                            i_episode, len(test_episode), test_episode.reward))

            i_episode += 1

        while render:
            render = self.render()

        return summary

    @abstractmethod
    def take_step(self, obs, action, episode, render=False):
        """Takes a step and updates
        
        Action action is executed and response is observed. Response is then
        used to update the value function. Data is stored in Episode object.

        Parameters
        ----------
        obs : :obj:`list` of :obj:`int` or :obj:`np.ndarray`
            observation
        action : :obj:`int`
            action
        episode : :obj:`Episode`
            Data container for all the episode data
        render : :obj:`bool`
            True when rendering every time-step (Default value = False)

        Returns
        -------
        :obj:`list` of :obj:`int` or `np.ndarray`
            next observation
        :obj:`int`
            next action
        :obj:`bool`
            done, True when obs_tp1 is terminal state
        :obj:`bool`
            render, True when rendering must continue
        """
        raise NotImplementedError

    def run_test_episode(self, i_episode):
        episode = Episode(i_episode)

        obs = self.env.reset()
        if self.test_render:
            self.env.render()
        for step in range(self.steps_episode):
            action = self._no_exploration(obs)
            obs, reward, done = self.env.step(action)
            episode.append(reward)

            if self.test_render:
                self.env.render()

            if done:
                break

        if self.test_render:
            self.env.render(close=True)

        return episode

    def get_action(self, obs):
        """Returns action for obs
        
        Return policy based on exploration strategy of the TDLearning object.
        
        When an exploration method is added make sure the method is added in
        the class attributes and below for ease of usage.

        Parameters
        ----------
        obs : :obj:`list` of :obj:`int` or `np.ndarray`
            observation

        Returns
        -------
        :obj:`int`
            next action

        Raises
        ------
        NotImplementedError
            Please choose from the available exploration methods, see class
            attributes.

        """
        if self.exploration_strategy == TDLearning.NONE:
            return self._no_exploration(obs)
        if self.exploration_strategy == TDLearning.E_GREEDY:
            return self._epsilon_greedy(obs)
        elif self.exploration_strategy == TDLearning.BOLTZMANN:
            return self._boltzmann(obs)
        else:
            raise NotImplementedError(
                "Please choose from the available exploration methods, "
                "see class attributes.")

    def _no_exploration(self, obs):
        """Policy without exploration strategy

        Parameters
        ----------
        obs : :obj:`list` of :obj:`int` or `np.ndarray`
            observation

        Returns
        -------
        :obj:`int`
            next action

        """
        q_values, actions = self.get_q_value(obs)
        _, max_action = max(zip(q_values, actions))
        return max_action

    def _epsilon_greedy(self, obs):
        """Policy with epsilon-greedy exploration method

        Parameters
        ----------
        obs : :obj:`list` of :obj:`int` or `np.ndarray`
            observation

        Returns
        -------
        :obj:`int`
            next action

        Raises
        ------
        Exception
            No maximum q-values were found.
        """
        q_values, actions = self.get_q_value(obs)

        if Scheduler in self.epsilon.__class__.__bases__:
            epsilon = self.epsilon.sample()
        else:
            epsilon = self.epsilon

        if random.random() < epsilon:
            return random.choice(actions)

        max_q = -float('inf')
        max_actions = []
        for action, q_value in zip(actions, q_values):
            if q_value > max_q:
                max_q = q_value
                max_actions = [action]
            elif q_value == max_q:
                max_actions.append(action)

        if not max_actions:
            raise Exception("No maximum q-values were found.")

        return random.choice(max_actions)

    def _boltzmann(self, obs):
        """Policy with Boltzmann exploration method

        Parameters
        ----------
        obs : :obj:`list` of :obj:`int` or `np.ndarray`
            observation

        Returns
        -------
        :obj:`int`
            next action
        """
        q_values, actions = self.get_q_value(obs)

        q_values = np.asarray(q_values)
        sum_q = np.sum(np.exp(q_values / self.temp))
        updated_values = [np.exp(q_value / self.temp) / sum_q for q_value in
                          q_values]

        return np.random.choice(actions, p=updated_values)

    @abstractmethod
    def render(self, *args, **kwargs):
        raise NotImplementedError

