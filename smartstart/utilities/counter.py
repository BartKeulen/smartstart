from collections import defaultdict

import numpy as np


class Counter:

    def __init__(self, state_action_shape):
        self.shape = state_action_shape
        self.visitation_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))

    def increment(self, state, action, next_state, value=1):
        self.visitation_counts[tuple(state)][action][tuple(next_state)] += value

    def get_visited_states(self):
        state_visitation_counts = self.get_state_visitation_counts()
        return np.asarray(np.where(state_visitation_counts > 0)).T

    def get_total_visitation_count(self):
        return np.sum(self.get_state_visitation_counts())

    def get_state_density_map(self):
        state_visitation_count = self.get_state_visitation_counts()
        return state_visitation_count / self.get_total_visitation_count()

    def get_state_visitation_counts(self, state=None):
        if state is None:
            state_visitation_counts = np.zeros(self.shape[:-1])
            for state_ in np.ndindex(self.shape[:-1]):
                state_visitation_counts[state_] = np.sum(self.get_state_action_visitation_counts(state_))
            return state_visitation_counts
        else:
            return np.sum(self.get_state_action_visitation_counts(state))

    def get_state_action_visitation_counts(self, state=None, action=None):
        if state is None:
            state_action_visitation_counts = np.zeros(self.shape)
            for state_action in np.ndindex(self.shape):
                visitation_counts, _ = self.get_state_action_state_visitation_counts(state_action[:-1], state_action[-1])
                state_action_visitation_counts[state_action] = np.sum(visitation_counts)
            return state_action_visitation_counts
        elif action is None:
            state_action_visitation_counts = np.zeros(self.shape[-1])
            for action_ in range(self.shape[-1]):
                visitation_counts, _ = self.get_state_action_state_visitation_counts(state, action_)
                state_action_visitation_counts[action_] = np.sum(visitation_counts)
            return state_action_visitation_counts
        else:
            visitation_counts, _ = self.get_state_action_state_visitation_counts(state, action)
            return np.sum(visitation_counts)

    def get_state_action_state_visitation_counts(self, state, action, next_state=None):
        if next_state is None:
            visitation_counts = list(self.visitation_counts[tuple(state)][action].values())
            next_states = list(self.visitation_counts[tuple(state)][action].keys())
            return visitation_counts, next_states
        else:
            return self.visitation_counts[tuple(state)][action][tuple(next_state)]

    def to_json_dict(self):
        json_dict = self.__dict__.copy()
        visitation_count_list = []
        for state, state_value in self.visitation_counts.items():
            state_visitation_count_list = {'state': (int(state[0]), int(state[1])),
                                           'value': []}
            for action, action_value in state_value.items():
                action_visitation_count_list = {'action': int(action),
                                                'value': []}
                for next_state, next_state_value in action_value.items():
                    next_state_visitation_count_list = {'next_state': (int(next_state[0]), int(next_state[1])),
                                                        'value': int(next_state_value)}
                    action_visitation_count_list['value'].append(next_state_visitation_count_list)
                state_visitation_count_list['value'].append(action_visitation_count_list)
            visitation_count_list.append(state_visitation_count_list)
        json_dict['visitation_counts'] = visitation_count_list
        return json_dict

    @classmethod
    def from_json_dict(cls, json_dict):
        json_dict = json_dict.copy()
        counter = Counter(json_dict['shape'])
        visitation_count_list = json_dict['visitation_counts']
        for state_value in visitation_count_list:
            for action_value in state_value['value']:
                for next_state_value in action_value['value']:
                    counter.increment(state_value['state'], action_value['action'], next_state_value['next_state'],
                                      next_state_value['value'])
        return counter