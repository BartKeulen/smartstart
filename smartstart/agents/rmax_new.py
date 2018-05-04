import logging
import pdb

import numpy as np
from smartstart.agents.counter import Counter
from smartstart.agents.valueiteration_new import ValueIteration

logger = logging.getLogger(__name__)


class RMax:

    def __init__(self,
                 vi,
                 m=2,
                 r_max=0.1,
                 tol=1e-4):
        self.name = self.__class__.__name__
        self.hash_table = vi.model.hash_table
        self.vi = vi
        self.m = m
        self.r_max = r_max
        self.tol = tol

        self.absorb_state = (-1, -1)
        self.s_prime = []
        self.C = []
        self.R_sum = []
        self.terminal_states = []

        self.update_counter = 0
        self.total_counter = 0

    def get_greedy_action(self, state):
        return self.vi.get_action(state)

    def get_action(self, state, use_ss_policy=False):
        s_hash = self.hash_table.hash_state(state)
        if s_hash >= len(self.C):
            return np.random.choice(self.vi.model.num_actions)
        state_action_visitation_counts = [sum(counts) for counts in self.C[s_hash]]
        actions_zero_count = [action_ for action_, count_ in enumerate(state_action_visitation_counts) if count_ == 0]
        if actions_zero_count:
            return np.random.choice(actions_zero_count)
        else:
            return self.vi.get_action(state)

    def update(self, state, action, reward, next_state, done):
        s_hash = self.hash_table.hash_state(state)
        ns_hash = self.hash_table.hash_state(next_state)

        # If next state terminal add to list of terminal states
        if done and ns_hash not in self.terminal_states:
            self.vi.model.add_terminal_state(next_state)

        # If new state add new count
        if s_hash >= self.vi.model.num_states:
            self.vi.model._add_state()
            self.s_prime.append([])
            self.C.append([])
            self.R_sum.append([])
            for _ in range(self.vi.model.num_actions):
                self.s_prime[-1].append([])
                self.C[-1].append([])
                self.R_sum[-1].append([])
        if ns_hash >= self.vi.model.num_states:
            self.vi.model._add_state()

        if ns_hash not in self.s_prime[s_hash][action]:
            self.s_prime[s_hash][action].append(ns_hash)
            self.C[s_hash][action].append(1)
            self.R_sum[s_hash][action].append(reward)
        else:
            idx = self.s_prime[s_hash][action].index(ns_hash)
            self.C[s_hash][action][idx] += 1
            self.R_sum[s_hash][action][idx] += reward

        r_old = sum(self.vi.model.get_reward(state, action))
        state_action_count = sum(self.C[s_hash][action])
        if state_action_count >= self.m:
            p_old = self.vi.model.get_transition(state, action, next_state)
            r_new = sum(self.R_sum[s_hash][action]) / state_action_count
            idx = self.s_prime[s_hash][action].index(ns_hash)
            p_new = self.C[s_hash][action][idx] / state_action_count
            if isinstance(p_old, list) or isinstance(p_new, list) or isinstance(r_old, list) or isinstance(r_new, list):
                pdb.set_trace()
            if r_old is None or p_old is None or abs(r_new - r_old) > self.tol or abs(p_new - p_old) > self.tol:
                for ns_idx, nss_hash in enumerate(self.s_prime[s_hash][action]):
                    p_new = self.C[s_hash][action][ns_idx] / state_action_count
                    self.vi.model.set(state, action, self.hash_table.un_hash_state(nss_hash), p_new, r_new)
                self.vi.optimize()
                self.update_counter += 1
        else:
            p_old = self.vi.model.get_transition(state, action, self.absorb_state)
            if r_old is None or p_old is None or abs(self.r_max - r_old) > self.tol or abs(p_old - 1) > self.tol:
                self.vi.model.set(state, action, self.absorb_state, 1, self.r_max)

                self.vi.optimize()
                self.update_counter += 1

        self.total_counter += 1

    def get_state_values(self):
        return self.vi.get_state_values()

    def get_state_action_values(self):
        return self.vi.get_state_action_values()

    # def to_json_dict(self):
    #     json_dict = self.__dict__.copy()
    #     json_dict['counter'] = self.counter.to_json_dict()
    #     json_dict['vi'] = self.vi.to_json_dict()
    #     json_dict['r_sum'] = self.r_sum.tolist()
    #     json_dict['state_list'] = [(int(state[0]), int(state[1])) for state in self.state_list]
    #     return json_dict
    #
    # @classmethod
    # def from_json_dict(cls, json_dict):
    #     json_dict = json_dict.copy()
    #     counter = Counter.from_json_dict(json_dict['counter'])
    #     vi = ValueIteration.from_json_dict(json_dict['vi'])
    #     rmax = cls(counter, vi, json_dict['m'], json_dict['r_max'])
    #     rmax.state_list = {(state[0], state[1]) for state in json_dict['state_list']}
    #     rmax.r_sum = np.asarray(json_dict['r_sum'])
    #     return rmax
