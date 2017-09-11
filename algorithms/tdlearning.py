import random
from abc import ABCMeta, abstractmethod

import numpy as np

from .counter import Counter
from utilities.datacontainers import Summary, Episode, SummarySmall


class TDLearning(Counter, metaclass=ABCMeta):
    NONE = 0
    E_GREEDY = 1
    BOLTZMANN = 2

    def __init__(self,
                 env,
                 num_episodes=1000,
                 max_steps=1000,
                 alpha=0.1,
                 gamma=0.99,
                 init_q_value=0.,
                 exploration=E_GREEDY,
                 epsilon=0.1,

                 seed=None):
        super(TDLearning, self).__init__(env)

        random.seed(seed)
        np.random.seed(seed)

        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.alpha = alpha
        self.gamma = gamma
        self.init_q_value = init_q_value
        self.Q = np.ones((self.env.w, self.env.h, self.env.num_actions)) * self.init_q_value
        self.exploration = exploration
        self.epsilon = epsilon
        self.seed = seed

    def reset(self):
        self.Q = np.ones((self.env.w, self.env.h, self.env.num_actions)) * self.init_q_value
        random.seed(self.seed)
        np.random.seed(self.seed)

    def get_q_values(self, obs):
        actions = self.env.possible_actions(obs)
        q_values = []
        for action in actions:
            idx = tuple(obs) + (action,)
            q_values.append(self.Q[idx])
        return q_values, actions

    def get_q_value(self, obs, action):
        idx = tuple(obs) + (action,)
        return self.Q[idx]

    @abstractmethod
    def get_next_q_action(self, obs_tp1, done):
        raise NotImplementedError("Use a subclass of TDLearning like QLearning or SARSA.")

    def update_q_value(self, obs, action, r, obs_tp1, done):
        next_q_value, action_tp1 = self.get_next_q_action(obs_tp1, done)
        td_error = self.alpha * (r + self.gamma * next_q_value - self.get_q_value(obs, action))

        idx = tuple(obs) + (action,)
        self.Q[idx] += self.alpha * td_error

        return self.Q[idx], action_tp1

    def train(self, render=False, render_episode=False, print_results=True):
        summary = SummarySmall(self.__class__.__name__ + "_" + self.env.name)

        for i_episode in range(self.num_episodes):
            episode = Episode()

            obs = self.env.reset()
            action = self.get_action(obs)

            for step in range(self.max_steps):
                obs, action, done, render = self.take_step(obs, action, episode, render)

                if done:
                    break

            # Render and/or print results
            message = "Episode: %d, steps: %d, reward: %.2f" % (i_episode, len(episode), episode.total_reward())
            if render or render_episode:
                value_map = self.Q.copy()
                value_map = np.max(value_map, axis=2)
                render_episode = self.env.render(value_map=value_map, density_map=self.get_density_map(),
                                                 message=message)
            if print_results:
                print("Episode: %d, steps: %d, reward: %.2f" % (i_episode, len(episode), episode.total_reward()))
            summary.append(episode)

        while render:
            value_map = self.Q.copy()
            value_map = np.max(value_map, axis=2)
            render = self.env.render(value_map=value_map, density_map=self.get_density_map())

        return summary

    def take_step(self, obs, action, episode, render=False):
        obs_tp1, r, done, _ = self.env.step(action)

        if render:
            value_map = self.Q.copy()
            value_map = np.max(value_map, axis=2)
            render = self.env.render(value_map=value_map, density_map=self.get_density_map())

        _, action_tp1 = self.update_q_value(obs, action, r, obs_tp1, done)

        self.increment(obs)

        episode.append(obs, action, r, obs_tp1, done)

        return obs_tp1, action_tp1, done, render

    def get_action(self, obs):
        if self.exploration == TDLearning.NONE:
            return self._no_exploration(obs)
        if self.exploration == TDLearning.E_GREEDY:
            return self._epsilon_greedy(obs)
        elif self.exploration == TDLearning.BOLTZMANN:
            return self._boltzmann(obs)
        else:
            raise NotImplementedError("Please choose from the available smartstart methods: E_GREEDY, BOLTZMANN.")

    def _no_exploration(self, obs):
        q_values, actions = self.get_q_values(obs)
        max_q, max_action = max(zip(q_values, actions))
        return max_action

    def _epsilon_greedy(self, obs):
        q_values, actions = self.get_q_values(obs)

        if random.random() < self.epsilon:
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
        q_values, actions = self.get_q_values(obs)

        sum_q = np.sum(np.exp(q_values))
        q_values = [np.exp(q_value) / sum_q for q_value in q_values]

        return np.random.choice(actions, p=q_values)

    def get_q_map(self):
        w, h = self.env.w, self.env.h

        q_map = np.zeros((w, h), dtype='int')
        for i in range(w):
            for j in range(h):
                obs = np.array([i, j])
                q_values, actions = self.get_q_values(obs)
                q_map[i, j] = int(max(q_values))

        return q_map


class TDLearningLambda(TDLearning):

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
        raise NotImplementedError("Use a subclass of TDLearningLambda like QLearningLambda or SARSALambda.")

    def update_q_value(self, obs, action, r, obs_tp1, done):
        next_q_value, action_tp1 = self.get_next_q_action(obs_tp1, done)
        td_error = self.alpha * (r + self.gamma * next_q_value - self.get_q_value(obs, action))

        idx = tuple(obs) + (action,)
        self.traces[idx] = 1
        active_traces = np.asarray(np.where(self.traces > self.threshold_traces))
        for i in range(active_traces.shape[1]):
            idx = tuple(active_traces[:, i])
            self.Q[idx] += self.alpha * td_error * self.traces[idx]
            self.traces[idx] *= self.gamma * self.lamb

        return self.Q[idx], action_tp1

    def train(self, render=False, render_episode=False, print_results=True):
        summary = Summary(self.__class__.__name__ + "_" + self.env.name)

        for i_episode in range(self.num_episodes):
            episode = Episode()

            obs = self.env.reset()
            action = self.get_action(obs)

            for step in range(self.max_steps):
                obs, action, done, render = self.take_step(obs, action, episode, render)

                if done:
                    break

            # Clear traces after episode
            self.traces = np.zeros((self.env.w, self.env.h, self.env.num_actions))

            # Render and/or print results
            message = "Episode: %d, steps: %d, reward: %.2f" % (i_episode, len(episode), episode.total_reward())
            if render or render_episode:
                value_map = self.Q.copy()
                value_map = np.max(value_map, axis=2)
                render_episode = self.env.render(value_map=value_map, density_map=self.get_density_map(),
                                                 message=message)
            if print_results:
                print("Episode: %d, steps: %d, reward: %.2f" % (i_episode, len(episode), episode.total_reward()))
            summary.append(episode)

        while render:
            value_map = self.Q.copy()
            value_map = np.max(value_map, axis=2)
            render = self.env.render(value_map=value_map, density_map=self.get_density_map())

        return summary