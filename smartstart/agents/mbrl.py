import logging
import pdb

import numpy as np

from smartstart.agents.counter import Counter
from smartstart.agents.valueiteration import ValueIteration


logger = logging.getLogger(__name__)


class MBRL:

    def __init__(self,
                 counter,
                 vi):
        self.name = self.__class__.__name__
        self.counter = counter
        self.vi = vi

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
        self.vi.r.set_reward(state, action, self.r_sum[state[0], state[1], action] / np.sum(visitation_counts))

        for visitation_count, next_state in zip(visitation_counts, next_states):
            self.vi.p.set_transition(state, action, next_state, visitation_count / np.sum(visitation_counts))

        if done and reward > 0:
            self.vi.terminal_states.append(tuple(next_state))

        # self.vi.state_list = list(self.state_list)
        self.vi.optimize()

    def get_state_values(self):
        return self.vi.get_state_values()

    def get_state_action_values(self):
        return self.vi.get_state_action_values()

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
        mbrl = cls(counter, vi)
        mbrl.state_list = {(state[0], state[1]) for state in json_dict['state_list']}
        mbrl.r_sum = np.asarray(json_dict['r_sum'])
        return mbrl