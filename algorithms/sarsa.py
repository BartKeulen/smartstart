import random

import numpy as np

from utilities import Summary, Episode
from smartstart.smartstart_file import SmartStart
from utilities.scheduler import LinearScheduler
from algorithms.counter import Counter


class SARSA(Counter):
    NONE = 0
    E_GREEDY = 1
    BOLTZMANN = 2

    def __init__(self, env,
                 num_episodes=100,
                 max_steps=1000,
                 alpha=0.1,
                 gamma=0.99,
                 init_q_value=0.,
                 exploration=E_GREEDY,
                 epsilon=0.1,
                 smart_start=None,
                 ss_scheduler=None,
                 seed=None):
        super(SARSA, self).__init__(env)
        random.seed(seed)
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.alpha = alpha
        self.gamma = gamma
        self.init_q_value = init_q_value
        self.Q = np.ones((self.env.w, self.env.h, self.env.num_actions))*self.init_q_value
        self.traces = None
        self.exploration = exploration
        self.epsilon = epsilon
        self.smart_start = smart_start
        if smart_start is not None and ss_scheduler is None:
            self.ss_scheduler = LinearScheduler(50, 100)
        else:
            self.ss_scheduler = ss_scheduler
        self.seed = seed

    def reset(self):
        self.Q = np.ones((self.env.w, self.env.h, self.env.num_actions))*self.init_q_value
        random.seed(self.seed)

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

    def update_q_value(self, obs, action, td_error):
        idx = tuple(obs) + (action,)
        self.Q[idx] += self.alpha * td_error
        return self.Q[idx]

    def train(self, render=False, render_episode=False, print_results=True):
        summary = Summary(self.__class__.__name__ + "_" + self.env.name)

        for i_episode in range(self.num_episodes):
            episode = Episode()

            obs = self.env.reset()
            init_obs = obs.copy()

            if self.smart_start:
                self.smart_start.policy_map.reset()

            max_steps = self.max_steps
            action = self.get_action(obs)
            if self.smart_start and i_episode > 0 and np.random.rand() <= self.ss_scheduler.sample():
                start_state, policy = self.smart_start.get_start(self)
                action = policy[0]
                for i in range(len(policy) - 1):
                    obs_tp1, r, done, _ = self.env.step(action)

                    if render:
                        value_map = self.Q.copy()
                        value_map = np.max(value_map, axis=2)
                        render = self.env.render(value_map=value_map, density_map=self.get_density_map())

                    if not done:
                        next_action = policy[i+1]
                        next_q_value = self.get_q_value(obs_tp1, next_action)
                    else:
                        next_q_value = 0.

                    td_error = r + self.gamma * next_q_value - self.get_q_value(obs, action)
                    self.update_q_value(obs, action, td_error)

                    self.increment(obs)
                    self.smart_start.policy_map.add_node(obs_tp1, action)

                    episode.append(obs, action, r, obs_tp1, done)

                    if done:
                        break

                    obs = obs_tp1
                    action = next_action
                max_steps = self.smart_start.exploration_steps

            for step in range(max_steps):
                obs_tp1, r, done, _ = self.env.step(action)

                if render:
                    value_map = self.Q.copy()
                    value_map = np.max(value_map, axis=2)
                    render = self.env.render(value_map=value_map, density_map=self.get_density_map())

                if not done:
                    next_action = self.get_action(obs_tp1)
                    next_q_value = self.get_q_value(obs_tp1, next_action)
                else:
                    next_q_value = 0.
                td_error = self.alpha * (r + self.gamma * next_q_value - self.get_q_value(obs, action))
                self.update_q_value(obs, action, td_error)

                self.increment(obs)

                if self.smart_start:
                    self.smart_start.policy_map.add_node(obs_tp1, action)

                episode.append(obs, action, r, obs_tp1, done)

                if done:
                    break

                obs = obs_tp1
                action = next_action

            print("Q-value start state: %s" % self.Q[tuple(init_obs)])
            if self.traces is not None:
                print("Trace start state: ", self.traces[tuple(init_obs)])

            if self.traces is not None:
                self.traces = np.zeros((self.env.w, self.env.h, self.env.num_actions))

            message = "Episode: %d, steps: %d, reward: %.2f" % (i_episode, len(episode), episode.total_reward())
            if render or render_episode:
                value_map = self.Q.copy()
                value_map = np.max(value_map, axis=2)
                render_episode = self.env.render(value_map=value_map, density_map=self.get_density_map(), message=message)
            if print_results:
                print("Episode: %d, steps: %d, reward: %.2f" % (i_episode, len(episode), episode.total_reward()))
            summary.append(episode)

        while render:
            value_map = self.Q.copy()
            value_map = np.max(value_map, axis=2)
            render = self.env.render(value_map=value_map, density_map=self.get_density_map())

        return summary

    def get_action(self, obs):
        if self.exploration == SARSA.NONE:
            return self._no_exploration(obs)
        if self.exploration == SARSA.E_GREEDY:
            return self._epsilon_greedy(obs)
        elif self.exploration == SARSA.BOLTZMANN:
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

        sum_q = np.exp(np.sum(q_values))
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


if __name__ == "__main__":
    from environments.gridworld import GridWorld, GridWorldVisualizer

    directory = '/home/bartkeulen/repositories/smartstart/data/tmp'

    np.random.seed()

    visualizer = GridWorldVisualizer()
    visualizer.add_visualizer(GridWorldVisualizer.LIVE_AGENT,
                              GridWorldVisualizer.CONSOLE,
                              GridWorldVisualizer.VALUE_FUNCTION,
                              GridWorldVisualizer.DENSITY)
    env = GridWorld.generate(GridWorld.EASY)
    env.visualizer = visualizer
    # env.wall_reset = True

    agent = SARSA(env, alpha=0.3, num_episodes=1500, max_steps=2500)

    summary = agent.train(render=False, render_episode=True)

    summary.save(directory=directory)

