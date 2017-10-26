"""Temporal-Difference module

Describes TDLearning and TDLearningLambda base classes for temporal
difference learning without and with eligibility traces.

See 'Reinforcement Learning: An Introduction by Richard S. Sutton and
Andrew G. Barto for more information.
"""
import random
from abc import ABCMeta, abstractmethod

import numpy as np
from smartstart.utilities.scheduler import Scheduler

from smartstart.utilities.datacontainers import Summary, Episode
from .counter import Counter


class TDLearning(Counter, metaclass=ABCMeta):
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
    COUNT_BASED = "Count-Based"
    UCT = "UCT"

    def __init__(self,
                 env,
                 num_episodes=1000,
                 max_steps=1000,
                 alpha=0.1,
                 gamma=0.99,
                 init_q_value=0.,
                 exploration=E_GREEDY,
                 epsilon=0.1,
                 temp=0.5,
                 beta=1.):
        super(TDLearning, self).__init__(env)

        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.alpha = alpha
        self.gamma = gamma
        self.init_q_value = init_q_value
        self.Q = np.ones(
            (self.env.w, self.env.h, self.env.num_actions)) * self.init_q_value
        self.exploration = exploration
        self.epsilon = epsilon
        self.temp = temp
        self.beta = beta

    def reset(self):
        """Resets Q-function
        
        The Q-function is set to the initial q-value for very state-action pair.
        """
        self.Q = np.ones(
            (self.env.w, self.env.h, self.env.num_actions)) * self.init_q_value

    def get_q_values(self, obs):
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
        actions = self.env.possible_actions(obs)
        q_values = []
        for action in actions:
            idx = tuple(obs) + (action,)
            q_values.append(self.Q[idx])
        return q_values, actions

    def get_q_value(self, obs, action):
        """Returns Q-value for obs-action pair

        Parameters
        ----------
        obs : :obj:`list` of :obj:`int` or :obj:`np.ndarray`
            observation
        action : int
            action

        Returns
        -------
        :obj:`float`
            Q-Value

        """
        idx = tuple(obs) + (action,)
        return self.Q[idx]

    @abstractmethod
    def get_next_q_action(self, obs_tp1, done):
        """Returns next Q-Value and next action
        
        Note:
            Has to be implemented in child class.

        Parameters
        ----------
        obs_tp1 : :obj:`list` of :obj:`int` or :obj:`np.ndarray`
            next observation
        done : bool
            True when obs_tp1 is terminal

        Raises
        ------
        NotImplementedError
            use a subclass of TDLearning like QLearning or SARSA.

        """
        raise NotImplementedError(
            "Use a subclass of TDLearning like QLearning or SARSA.")

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
        td_error = self.alpha * (
            reward + self.gamma * next_q_value - self.get_q_value(obs, action))

        idx = tuple(obs) + (action,)
        self.Q[idx] += self.alpha * td_error

        return self.Q[idx], action_tp1

    def train(self, render=False, render_episode=False, print_results=True):
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
        summary = Summary(self.__class__.__name__ + "_" + self.env.name)

        for i_episode in range(self.num_episodes):
            episode = Episode()

            obs = self.env.reset()
            action = self.get_action(obs)

            for _ in range(self.max_steps):
                obs, action, done, render = self.take_step(obs, action, episode,
                                                           render)

                if done:
                    break

            # Render and/or print results
            message = "Episode: %d, steps: %d, reward: %.2f" % (
                i_episode, len(episode), episode.average_reward())
            if render or render_episode:
                value_map = self.Q.copy()
                value_map = np.max(value_map, axis=2)
                render_episode = self.env.render(value_map=value_map,
                                                 density_map=self.get_density_map(),
                                                 message=message)

            if print_results:
                print("Episode: %d, steps: %d, reward: %.2f" % (
                    i_episode, len(episode), episode.average_reward()))
            summary.append(episode)

        while render:
            value_map = self.Q.copy()
            value_map = np.max(value_map, axis=2)
            render = self.env.render(value_map=value_map,
                                     density_map=self.get_density_map())

        return summary

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
        obs_tp1, reward, done, _ = self.env.step(action)

        if render:
            value_map = self.Q.copy()
            value_map = np.max(value_map, axis=2)
            render = self.env.render(value_map=value_map,
                                     density_map=self.get_density_map())

        _, action_tp1 = self.update_q_value(obs, action, reward, obs_tp1, done)

        self.increment(obs, action, obs_tp1)

        episode.append(obs, action, reward, obs_tp1, done)

        return obs_tp1, action_tp1, done, render

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
        if self.exploration == TDLearning.NONE:
            return self._no_exploration(obs)
        if self.exploration == TDLearning.E_GREEDY:
            return self._epsilon_greedy(obs)
        elif self.exploration == TDLearning.BOLTZMANN:
            return self._boltzmann(obs)
        elif self.exploration == TDLearning.COUNT_BASED:
            return self._count_based(obs)
        elif self.exploration == TDLearning.UCT:
            return self._uct(obs)
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
        q_values, actions = self.get_q_values(obs)
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
        q_values, actions = self.get_q_values(obs)

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
        q_values, actions = self.get_q_values(obs)

        q_values = np.asarray(q_values)
        sum_q = np.sum(np.exp(q_values / self.temp))
        updated_values = [np.exp(q_value / self.temp) / sum_q for q_value in
                          q_values]

        return np.random.choice(actions, p=updated_values)

    def _count_based(self, obs):
        """Policy with count-based exploration method

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
        q_values, actions = self.get_q_values(obs)
        tot_count = self.get_count(obs)

        max_value = -float('inf')
        max_actions = []
        for action, value in zip(actions, q_values):
            value += self.beta * tot_count / (self.get_count(obs, action) + 1e-12)
            if value > max_value:
                max_value = value
                max_actions = [action]
            elif value == max_value:
                max_actions.append(action)

        if not max_actions:
            raise Exception("No maximum q-values were found.")

        return random.choice(max_actions)

    def _uct(self, obs):
        """Policy with UCT exploration method

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
        q_values, actions = self.get_q_values(obs)
        tot_count = self.get_count(obs)

        max_value = -float('inf')
        max_actions = []
        for action, value in zip(actions, q_values):
            if tot_count > 0:
                count = self.get_count(obs, action)
                if count == 0:
                    value = np.inf
                else:
                    value += 2 * self.beta * np.sqrt(np.log(tot_count) / np.log(self.get_count(obs, action)))
            if value > max_value:
                max_value = value
                max_actions = [action]
            elif value == max_value:
                max_actions.append(action)

        if not max_actions:
            raise Exception("No maximum q-values were found.")

        return random.choice(max_actions)

    def get_q_map(self):
        """Returns value map for environment
        
        Value-map will be a numpy array equal to the width and height
        of the environment. Each entry (state) will hold the maximum
        action-value function associated with that state.

        Returns
        -------
        :obj:`np.ndarray`
            value map

        """
        width, height = self.env.w, self.env.h

        q_map = np.zeros((width, height), dtype='int')
        for i in range(width):
            for j in range(height):
                obs = np.array([i, j])
                q_values, _ = self.get_q_values(obs)
                q_map[i, j] = int(max(q_values))

        return q_map


class TDLearningLambda(TDLearning):
    """Base class for temporal difference methods using eligibility traces
    
    Child class of TDLearning, update_q_value and train method are modified
    for using eligibility traces.

    Parameters
    ----------
    env : :obj:`Environment`
        environment
    lamb : :obj:`float`
        eligibility traces decay parameter
    threshold_traces : :obj:`float`
        threshold for activation of trace
    *args :
        see parent class TDLearning
    **kwargs :
        see parent class TDLearning

    Attributes
    ----------
    lamb : :obj:`float`
        eligibility traces decay parameter
    threshold_traces : :obj:`float`
        threshold for activation of trace
    traces : :obj:`np.ndarray`
        numpy ndarray holding the traces for each state-action pair

    """
    def __init__(self,
                 env,
                 lamb=0.75,
                 threshold_traces=1e-3,
                 *args, **kwargs):
        super(TDLearningLambda, self).__init__(env, *args, **kwargs)
        self.lamb = lamb
        self.threshold_traces = threshold_traces
        self.traces = np.zeros((self.env.w, self.env.h, self.env.num_actions))

    @abstractmethod
    def get_next_q_action(self, obs_tp1, done):
        """Returns next Q-Value and action
        
        Note:
            Has to be implemented in child class.

        Parameters
        ----------
        obs_tp1 : :obj:`list` of :obj:`int` or `np.ndarray`
            next observation
        done : :obj:`bool`
            True when obs_tp1 is terminal

        Raises
        ------
            Use a subclass of TDLearning like
            QLearningLambda or SARSALambda.

        """
        raise NotImplementedError(
            "Use a subclass of TDLearningLambda like QLearningLambda or "
            "SARSALambda.")

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
        next_q_value, action_tp1 = self.get_next_q_action(obs_tp1, done)
        td_error = self.alpha * (
            reward + self.gamma * next_q_value - self.get_q_value(obs, action))

        idx = tuple(obs) + (action,)
        self.traces[idx] = 1
        active_traces = np.asarray(
            np.where(self.traces > self.threshold_traces))
        for i in range(active_traces.shape[1]):
            idx = tuple(active_traces[:, i])
            self.Q[idx] += self.alpha * td_error * self.traces[idx]
            self.traces[idx] *= self.gamma * self.lamb

        return self.Q[idx], action_tp1

    def train(self, render=False, render_episode=False, print_results=True):
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
        :obj:`Summary`
            Summary Object containing the training data

        """
        summary = Summary(self.__class__.__name__ + "_" + self.env.name)

        for i_episode in range(self.num_episodes):
            episode = Episode()

            obs = self.env.reset()
            action = self.get_action(obs)

            for _ in range(self.max_steps):
                obs, action, done, render = self.take_step(obs, action, episode,
                                                           render)

                if done:
                    break

            # Clear traces after episode
            self.traces = np.zeros(
                (self.env.w, self.env.h, self.env.num_actions))

            # Render and/or print results
            message = "Episode: %d, steps: %d, reward: %.2f" % (
                i_episode, len(episode), episode.total_reward())
            if render or render_episode:
                value_map = self.Q.copy()
                value_map = np.max(value_map, axis=2)
                render_episode = self.env.render(value_map=value_map,
                                                 density_map=self.get_density_map(),
                                                 message=message)
            if print_results:
                print("Episode: %d, steps: %d, reward: %.2f" % (
                    i_episode, len(episode), episode.total_reward()))
            summary.append(episode)

        while render:
            value_map = self.Q.copy()
            value_map = np.max(value_map, axis=2)
            render = self.env.render(value_map=value_map,
                                     density_map=self.get_density_map())

        return summary
