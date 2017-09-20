import random
from collections import defaultdict

import numpy as np

from algorithms.counter import Counter
from algorithms.valueiteration import ValueIteration
from utilities.datacontainers import Summary, Episode


class RMax(Counter):

    def __init__(self,
                 env,
                 num_episodes=1000,
                 max_steps=1000,
                 gamma=0.99,
                 m=5,
                 min_error=0.01,
                 max_itr=100,
                 R_max=1.,
                 seed=None):
        super(RMax, self).__init__(env)
        random.seed(seed)
        np.random.seed(seed)

        self.env = env
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.gamma = gamma
        self.m = m
        self.min_error = min_error
        self.max_itr = max_itr
        self.R_max = R_max

        self.policy = ValueIteration(self.env, gamma, min_error, max_itr)
        self.R_sum = defaultdict(lambda: defaultdict(lambda: 0))

        self.obs_abs = "o"
        for action in env.possible_actions(self.obs_abs):
            self.policy.set_reward(self.obs_abs, action, self.obs_abs, self.R_max)
            self.policy.set_transition(self.obs_abs, action, self.obs_abs, 1)

    def train(self, render=False, render_episode=False, print_results=True):
        summary = Summary(self.__class__.__name__ + "_" + self.env.name)

        for i_episode in range(self.num_episodes):
            episode = Episode()

            obs = self.env.reset()
            self.policy.add_obs(obs)
            action = self.policy.get_action(obs)

            for step in range(self.max_steps):
                obs, action, done, render = self.take_step(obs, action, episode, render)

                if done:
                    break

            # Render and/or print results
            message = "Episode: %d, steps: %d, reward: %.2f" % (i_episode, len(episode), episode.total_reward())
            if render or render_episode:
                render_episode = self.env.render(value_map=self.get_value_map(),
                                                 density_map=self.get_density_map(),
                                                 message=message)

            if print_results:
                print(
                    "Episode: %d, steps: %d, reward: %.2f" % (i_episode, len(episode), episode.total_reward()))
            summary.append(episode)

        while render or render_episode:
            render = self.env.render(value_map=self.get_value_map(),
                                     density_map=self.get_density_map())
            render_episode = render

        return summary

    def take_step(self, obs, action, episode, render=False):
        obs_tp1, r, done, _ = self.env.step(action)
        self.policy.add_obs(obs_tp1)

        if render:
            render = self.env.render()

        self.increment(obs, action, obs_tp1)
        self.R_sum[tuple(obs) + (action,)][tuple(obs_tp1)] += r

        if self.get_count(obs, action) > self.m:
            # Known state
            for obs_prime, count in self.count_map[tuple(obs)][(action,)].items():
                reward = self.R_sum[tuple(obs) + (action,)][tuple(obs_prime)] / self.get_count(obs, action)
                self.policy.set_reward(obs, action, obs_prime, reward)
                self.policy.set_transition(obs, action, obs_prime, count / self.get_count(obs, action))
        # TODO: R-Max setting unknown state optimistically not working
        # else:
        #     # Unknown state
        #     self.policy.set_reward(obs, action, self.R_max)
        #     self.policy.set_transition(obs, action, self.obs_abs, 1)

        self.policy.optimize()

        episode.append(obs, action, r, obs_tp1, done)

        action_tp1 = self.policy.get_action(obs_tp1)

        return obs_tp1, action_tp1, done, render

    def get_value_map(self):
        return self.policy.get_value_map()


if __name__ == "__main__":
    from environments.gridworld import GridWorld, GridWorldVisualizer

    directory = '/home/bartkeulen/repositories/smartstart/data/tmp'

    np.random.seed()

    visualizer = GridWorldVisualizer()
    visualizer.add_visualizer(GridWorldVisualizer.LIVE_AGENT,
                              GridWorldVisualizer.DENSITY,
                              GridWorldVisualizer.VALUE_FUNCTION,
                              GridWorldVisualizer.CONSOLE)
    env = GridWorld.generate(GridWorld.EASY)
    env.visualizer = visualizer
    # env.wall_reset = True

    agent = RMax(env,
                 num_episodes=1000,
                 max_steps=1000,
                 m=1)

    summary = agent.train(render_episode=True)

    summary.save(directory=directory)