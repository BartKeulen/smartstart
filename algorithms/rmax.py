import random
from collections import defaultdict

import numpy as np

from utilities.datacontainers import Summary, Episode


class RMax(object):

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

        self.obs_abs = None

        self.count = defaultdict(lambda: defaultdict(lambda: 0))
        self.transition = defaultdict(lambda: defaultdict(lambda: 0))
        self.R_sum = defaultdict(lambda: 0)
        self.R = defaultdict(lambda: 0)
        self.T = defaultdict(lambda: 0)
        self.V = defaultdict(lambda: 0)
        self.obses = set()

    def increment_count(self, obs, action, obs_tp1):
        self.count[tuple(obs) + (action, )][tuple(obs_tp1)] += 1

    def get_count(self, obs, action, obs_tp1=None):
        if obs_tp1 is not None:
            return self.count[tuple(obs) + (action, )][tuple(obs_tp1)]

        return self.count[tuple(obs) + (action, )].values()

    def set_transition(self, obs, action, obs_tp1, value):
        self.transition[tuple(obs) + (action, )][tuple(obs_tp1)] = value

    def get_transition(self, obs, action, obs_tp1=None):
        if obs_tp1 is not None:
            return self.transition[tuple(obs) + (action, )][tuple(obs_tp1)]

        return self.transition[tuple(obs) + (action, )]

    def train(self, render=False, render_episode=False, print_results=True):
        summary = Summary(self.__class__.__name__ + "_" + self.env.name)

        for i_episode in range(self.num_episodes):
            episode = Episode()

            obs = self.env.reset()
            self.obses.add(tuple(obs))
            action = self.get_action(obs)

            for step in range(self.max_steps):
                obs, action, done, render = self.take_step(obs, action, episode, render)

                if done:
                    break

            # Render and/or print results
            message = "Episode: %d, steps: %d, reward: %.2f" % (i_episode, len(episode), episode.average_reward())
            if render or render_episode:
                render_episode = self.env.render()

            if print_results:
                print(
                    "Episode: %d, steps: %d, reward: %.2f" % (i_episode, len(episode), episode.average_reward()))
            summary.append(episode)

        while render:
            render = self.env.render()

        return summary

    def take_step(self, obs, action, episode, render=False):
        obs_tp1, r, done, _ = self.env.step(action)
        self.obses.add(tuple(obs_tp1))

        if render:
            render = self.env.render()

        self.increment_count(obs, action, obs_tp1)
        self.R_sum[tuple(obs) + (action,)] += r

        if sum(self.get_count(obs, action)) > self.m:
            # Known state
            self.R[tuple(obs) + (action,)] = self.R_sum[tuple(obs) + (action,)]/sum(self.get_count(obs, action))
            for obs_prime, count in self.get_count(obs, action):
                self.set_transition(obs, action, obs_prime, count/sum(self.get_count(obs, action)))
        else:
            # Unknown state
            self.R[tuple(obs), (action,)] = self.R_max
            self.T[tuple(obs), (action,)][tuple(self.obs_abs)] = 1

        self.value_iteration()

        episode.append(obs, action, r, obs_tp1, done)

        return obs_tp1, action, done, render

    def value_iteration(self):
        itr = 0
        while True:
            itr += 1
            delta = 0
            for obs in self.obses:
                v = self.V[obs]
                v_new = []
                for action in self.env.possible_actions(obs):
                    r = self.R[tuple(obs) + (action,)]
                    v_tmp = 0.
                    for obs_prim, transition in self.get_transition(obs, action).items():
                        v_tmp += transition * (r + self.gamma * self.V[obs_prim])
                    v_new.append(v_tmp)
                self.V[obs] = max(v_new)
                delta = max(delta, abs(v - self.V[obs]))
            if delta < self.min_error or itr >= self.max_itr:
                break

    def get_action(self, obs):
        actions = self.env.possible_actions(obs)
        max_actions = []
        max_value = -float('inf')
        for action in actions:
            r = self.R[tuple(obs) + (action,)]
            value = 0.
            for obs_prime, transition in self.get_transition(obs, action).items():
                value += transition * (r + self.gamma * self.V[obs_prime])
            if value > max_value:
                max_value = value
                max_actions = [action]
            elif value == max_value:
                max_actions.append(action)

        return random.choice(max_actions)