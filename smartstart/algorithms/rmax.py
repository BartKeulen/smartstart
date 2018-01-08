import random
from collections import defaultdict

import numpy as np

from smartstart.algorithms.dynamicprogramming import ValueIteration
from smartstart.algorithms.counter import Counter
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


class SmartStartRMax(RMax):

    def __init__(self,
                 env,
                 c_ss=2.,
                 eta=0.5,
                 m=5,
                 vi_gamma=0.999,
                 vi_min_error=1e-5,
                 vi_max_itr=1000,
                 *args,
                 **kwargs):
        super(SmartStartRMax, self).__init__(env, *args, **kwargs)
        self.c_ss = c_ss
        self.eta = eta
        self.m = m

        self.policy = ValueIteration(self.env, vi_gamma, vi_min_error, vi_max_itr)

    def get_start(self):
        """Determines the smart start state

        The smart start is determined using the UCB1 algorithm. The UCB1
        algorithm is a well known exploration strategy for multi-arm
        bandit problems. The smart start is chosen according to

        :math:`smart_start = \arg\max\limits_s\left(alpha * \max\limits_a
        Q(s,a) + \sqrt{\frac{beta * \log\sum\limits_{s'} C(s'}{C(s}} \right)`

        Where
            * :math:`\alpha` = exploitation_param
            * :math:`\beta` = exploration_param

        Returns
        -------
        :obj:`np.ndarray`
            smart start
        """
        count_map = self.get_count_map()
        if count_map is None:
            return None
        possible_starts = np.asarray(np.where(count_map > 0))
        if not possible_starts.any():
            return None

        smart_start = None
        max_ucb = -float('inf')
        for i in range(possible_starts.shape[1]):
            obs = possible_starts[:, i]
            value = self.vi.V[tuple(obs)]
            ucb = value + np.sqrt((self.c_ss * np.log(np.sum(count_map))) / count_map[tuple(obs)])
            if ucb > max_ucb:
                smart_start = obs
                max_ucb = ucb
        return smart_start

    def train(self, test_freq=0, render=False, render_episode=False, print_results=True):
        summary = Summary(self.__class__.__name__, self.env.name)

        i_episode = 0
        total_steps = 0
        while total_steps < self.max_steps:
            episode = Episode(i_episode)

            obs = self.env.reset()

            # eta probability of using smart start
            self.use_smart_start_policy = False
            finished = False
            if i_episode > 0 and np.random.rand() <= self.eta:
                # Step 1: Choose smart start
                start_state = self.get_start()

                # Step 2: Guide to smart start
                self.dynamic_programming(start_state)

                finished = False

                # Set using smart start policy to True
                self.use_smart_start_policy = True
                for i in range(self.steps_episode):
                    action = self.policy.get_action(obs)

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

                    if np.array_equal(obs, start_state):
                        self.vi.optimize()
                        break

                    if done:
                        self.vi.optimize()
                        finished = True
                        break

            # Turn using smart start policy off
            self.use_smart_start_policy = False

            # Perform normal reinforcement learning
            if not finished:
                for _ in range(self.steps_episode - len(episode)):
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

    def dynamic_programming(self, start_state):
        """Fits transition model, reward function and performs dynamic
        programming

        Transition model is fitted using the following equation

            :math:`T(s,a,s') = \frac{C(s,a,s'}{C(s,a)}`

        Where C(*) is the visitation count. The reward function is zero
        everywhere except for the transition that results in the smart start

            :math:`R(s,a,s') = 1 \text{if} s' == s_{ss}`
            :math:`R(s,a,s') = 0 \text{otherwise}`

        Dynamic programming is done using value iteration.

        Parameters
        ----------
        start_state : :obj:`np.ndarray`
            SmartStart state

        """
        # Reset policy
        self.policy.reset()

        # Dictionary for reverse searching transitions for optimal observation list
        transitions = defaultdict(lambda: [])

        # Fit transition model and reward function
        for obs_c, obs_count in self.count_map.items():
            for action, action_count in obs_count.items():
                for obs_tp1, count in action_count.items():

                    transitions[obs_tp1].append(obs_c)

                    if obs_tp1 == tuple(start_state):
                        self.policy.R[obs_c + action] = 1.
                        self.policy.goal = obs_tp1
                    self.policy.T[obs_c + action][obs_tp1] = \
                        count / sum(self.count_map[obs_c][action].
                                    values())

        # Get optimal observation list for dynamic programming
        obses = []
        remaining_obses = [tuple(start_state)]
        while remaining_obses:
            next_obs = remaining_obses.pop(0)
            obses.append(next_obs)
            next_obses = transitions[next_obs]
            for obs in next_obses:
                if obs not in obses and obs not in remaining_obses:
                    remaining_obses.append(obs)

        self.policy.obses = obses

        # Perform dynamic programming
        self.policy.optimize()