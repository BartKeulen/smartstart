import logging
import pdb

import numpy as np
from smartstart.agents.counter import Counter
from smartstart.agents.valueiteration import ValueIteration

logger = logging.getLogger(__name__)


class RMax:

    def __init__(self,
                 counter,
                 vi,
                 m=2,
                 r_max=0.1):
        self.name = self.__class__.__name__
        self.counter = counter
        self.vi = vi
        self.m = m
        self.r_max = r_max

        self.absorb_state = (-1, -1)
        self.state_list = set()
        self.r_sum = np.zeros(counter.shape)

    def get_greedy_action(self, state):
        return self.vi.get_action(state)

    def get_action(self, state, use_ss_policy=False):
        state_action_visitation_counts = self.counter.get_state_action_visitation_counts(state)
        actions_zero_count = [action_ for action_, count_ in enumerate(state_action_visitation_counts) if count_ == 0]
        if actions_zero_count:
            return np.random.choice(actions_zero_count)
        else:
            return self.vi.get_action(state)

    def update(self, state, action, reward, next_state, done):
        self.counter.increment(state, action, next_state)
        self.state_list.add(tuple(state))

        self.r_sum[state[0], state[1], action] += reward

        visitation_counts, next_states = self.counter.get_state_action_state_visitation_counts(state, action)
        if np.sum(visitation_counts) >= self.m:
            self.vi.r.set_reward(state, action, self.r_sum[state[0], state[1], action] / np.sum(visitation_counts))

            self.vi.p.clear_transitions(state, action)
            for visitation_count, next_state in zip(visitation_counts, next_states):
                self.vi.p.set_transition(state, action, next_state, visitation_count / np.sum(visitation_counts))
        else:
            self.vi.r.set_reward(state, action, self.r_max)
            self.vi.p.set_transition(state, action, self.absorb_state, 1.)

        if done:
            self.vi.terminal_states.append(tuple(next_state))

        self.vi.state_list = list(self.state_list)
        self.vi.optimize()

    def get_state_values(self):
        return self.vi.get_state_values()

    def to_json_dict(self):
        json_dict = self.__dict__.copy()
        json_dict['counter'] = self.counter.to_json_dict()
        json_dict['vi'] = self.vi.to_json_dict()
        json_dict['r_sum'] = self.r_sum.tolist()
        json_dict['state_list'] = [(int(state[0]), int(state[1])) for state in self.state_list]
        return json_dict

    @classmethod
    def from_json_dict(cls, json_dict):
        json_dict = json_dict.copy()
        counter = Counter.from_json_dict(json_dict['counter'])
        vi = ValueIteration.from_json_dict(json_dict['vi'])
        rmax = cls(counter, vi, json_dict['m'], json_dict['r_max'])
        rmax.state_list = {(state[0], state[1]) for state in json_dict['state_list']}
        rmax.r_sum = np.asarray(json_dict['r_sum'])
        return rmax
