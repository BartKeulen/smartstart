import random
from collections import defaultdict

from smartstart.algorithms.dynamicprogramming import ValueIteration
from smartstart.algorithms.counter import Counter
from smartstart.environments.gridworld import GridWorld
from smartstart.environments.gridworldvisualizer import GridWorldVisualizer
from smartstart.utilities.datacontainers import Summary, Episode


class RMax(Counter):

    def __init__(self, env, r_max=100, m=5, max_steps=500000, steps_episode=1000, *args, **kwargs):
        super(RMax, self).__init__(env, *args, **kwargs)
        self.vi = ValueIteration(env, *args, **kwargs)
        self.m = m
        self.r_max = r_max
        self.max_steps = max_steps
        self.steps_episode = steps_episode
        self.s_abs = 's'
        self.r_sum = defaultdict(lambda: 0)

        self.test_render = False

    def get_action(self, obs):
        actions = self.env.possible_actions(obs)
        new_actions = []
        for action in actions:
            if self.get_count(obs, action) == 0:
                new_actions.append(action)
        if new_actions:
            return random.choice(new_actions)

        return self.vi.get_action(obs)

    def train(self, test_freq=0, render=False, render_episode=False, print_results=True):
        summary = Summary(self.__class__.__name__, self.env.name)

        i_episode = 0
        total_steps = 0
        while total_steps < self.max_steps:
            episode = Episode(i_episode)

            obs = self.env.reset()

            for _ in range(self.steps_episode):
                action = self.get_action(obs)
                self.vi.obses.add(tuple(obs))
                obs_tp1, reward, done = self.env.step(action)
                total_steps += 1

                if render:
                    render = self.env.render()

                self.increment(obs, action, obs_tp1)
                self.r_sum[tuple(obs) + (action,)] += reward

                episode.append(reward)

                current_count = self.get_count(obs, action)
                if current_count >= self.m:
                    reward_function = self.r_sum[tuple(obs) + (action,)] / current_count
                    self.vi.set_reward(obs, action, reward_function)

                    self.vi.T[tuple(obs) + (action,)].clear()
                    for next_obs in self.next_obses(obs, action):
                        transition_model = self.get_count(obs, action, next_obs) / current_count
                        self.vi.set_transition(obs, action, next_obs, transition_model)
                else:
                    self.vi.set_reward(obs, action, self.r_max)
                    self.vi.set_transition(obs, action, self.s_abs, 1)

                self.vi.optimize()

                if done:
                    break

                obs = obs_tp1

            # Add training episode to summary
            summary.append(episode)

            # Render and/or print results
            message = "Episode: %d, steps: %d, reward: %.2f" % (
                i_episode, len(episode), episode.reward)
            if render or render_episode:
                render_episode = self.env.render(message=message)

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
            render = self.env.render()

        return summary

    def run_test_episode(self, i_episode):
        episode = Episode(i_episode)

        obs = self.env.reset()
        if self.test_render:
            self.env.render()
        for step in range(self.steps_episode):
            action = self.vi.get_action(obs)
            obs, reward, done = self.env.step(action)
            episode.append(reward)

            if self.test_render:
                self.env.render()

            if done:
                break

        if self.test_render:
            self.env.render(close=True)

        return episode
