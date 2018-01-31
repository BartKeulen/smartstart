import logging

import numpy as np
import pdb
from collections import defaultdict

logger = logging.getLogger(__name__)


class ValueIteration:

    def __init__(self,
                 state_action_shape,
                 transition_model=None,
                 reward_function=None,
                 terminal_states=None,
                 state_list=None,
                 gamma=0.99,
                 min_error=1e-3,
                 max_itr=1000):
        self.state_action_shape = state_action_shape
        self.state_values = np.zeros(state_action_shape[:-1])
        if transition_model is None:
            transition_model = TransitionModel(list(range(state_action_shape[-1])))
        if reward_function is None:
            reward_function = RewardFunction(list(range(state_action_shape[-1])))
        self.p = transition_model
        self.r = reward_function

        if state_list is None:
            state_list = np.asarray(list(np.ndindex(self.state_values.shape)))
        self.state_list = state_list
        if terminal_states is None:
            terminal_states = []
        self.terminal_states = terminal_states

        self.gamma = gamma
        self.min_error = min_error
        self.max_itr = max_itr

    def set(self, transition_model, reward_function, terminal_states=None, state_list=None, reset_state_values=True):
        if reset_state_values:
            self.state_values.fill(0)
        self.p = transition_model
        self.r = reward_function
        if state_list is None:
            state_list = np.asarray(list(np.ndindex(self.state_values.shape)))
        self.state_list = state_list
        if terminal_states is None:
            terminal_states = []
        self.terminal_states = terminal_states

    def get_action(self, state):
        values = self.calc_state_action_values(state)
        return np.argmax(values)

    def optimize(self):
        if self.p is None or self.r is None:
            raise NotImplementedError('Please initialize the transition model p and reward function r')

        if self.state_list is None:
            self.state_list = np.asarray(list(np.ndindex(self.state_values.shape)))

        delta = 0
        for itr in range(self.max_itr):
            delta = 0
            for state in self.state_list:
                if tuple(state) not in self.terminal_states:
                    cur_state_value = self.state_values[state[0], state[1]]

                    values = self.calc_state_action_values(state)
                    new_state_value = max(values)
                    self.state_values[state[0], state[1]] = new_state_value
                    delta = max(delta, abs(cur_state_value - new_state_value))
            if delta < self.min_error:
                logger.debug("VI Converged in %d iterations, delta: %.7f" % (itr, delta))
                return
        logger.debug("VI did not converge, delta: %.7f" % delta)

    def calc_state_action_values(self, state):
        rewards = self.r.get_reward(state)
        values = []
        for transition_probs, next_states, reward in zip(*self.p.get_transition(state), rewards):
            next_state_values = [0. if next_state in self.terminal_states else
                                 self.state_values[next_state[0], next_state[1]] for next_state in next_states]

            values.append(sum([transition_prob * (reward + self.gamma * next_state_value) for
                               transition_prob, next_state_value in
                               zip(transition_probs, next_state_values)]))
        return values

    def get_state_action_values(self):
        state_action_values = np.zeros(self.state_action_shape)
        for i in range(self.state_action_shape[0]):
            for j in range(self.state_action_shape[1]):
                state_action_values[i, j] = self.calc_state_action_values([i, j])
        return state_action_values

    def get_state_values(self):
        return self.state_values

    def to_json_dict(self):
        json_dict = self.__dict__.copy()
        json_dict['state_values'] = self.state_values.tolist()
        json_dict['p'] = self.p.to_json_dict()
        json_dict['r'] = self.r.to_json_dict()
        state_list = np.asarray(self.state_list)
        json_dict['state_list'] = state_list.tolist()
        terminal_states = np.asarray(self.terminal_states)
        json_dict['terminal_states'] = terminal_states.tolist()
        return json_dict

    @classmethod
    def from_json_dict(cls, json_dict):
        json_dict = json_dict.copy()
        state_values = np.asarray(json_dict['state_values'])
        json_dict['transition_model'] = TransitionModel.from_json_dict(json_dict['p'])
        json_dict['reward_function'] = RewardFunction.from_json_dict(json_dict['r'])
        del json_dict['state_values']
        del json_dict['p']
        del json_dict['r']
        vi = cls(**json_dict)
        vi.state_values = state_values
        return vi


class TransitionModel:

    def __init__(self, actions):
        self.actions = actions
        self.P = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))

    def set_transition(self, state, action, next_state, transition_prob):
        self.P[tuple(state)][action][tuple(next_state)] = transition_prob

    def add_transition(self, state, action, next_state, transition_prob):
        self.P[tuple(state)][action][tuple(next_state)] += transition_prob

    def get_transition(self, state, action=None, next_state=None):
        if action is not None:
            assert action in self.actions
        if action is None:
            next_states = [list(self.P[tuple(state)][action_].keys()) for action_ in self.actions]
            transition_probs = [list(self.P[tuple(state)][action_].values()) for action_ in self.actions]
            return transition_probs, next_states
        elif next_state is None:
            next_states = list(self.P[tuple(state)][action].keys())
            transition_probs = list(self.P[tuple(state)][action].values())
            return transition_probs, next_states
        else:
            transition_prob = self.P[tuple(state)][action][tuple(next_state)]
            return transition_prob, next_state

    def clear_transitions(self, state=None, action=None, next_state=None):
        if state is None:
            self.P.clear()
        elif action is None:
            self.P[tuple(state)].clear()
        elif next_state is None:
            self.P[tuple(state)][action].clear()

    def validate(self):
        for state, action_next_states_transitions in self.P.items():
            for action, next_state_transitions in action_next_states_transitions.items():
                total_prob = sum(list(next_state_transitions.values()))
                if abs(total_prob - 1) > 1e-5:
                    raise ValueError('Total probability for state %s and action %s do not add up to 1. Total probability is %.2f' % (state, action, total_prob))

    def to_json_dict(self):
        json_dict = self.__dict__.copy()
        transitions = []
        for state, state_value in self.P.items():
            state_transitions = {'state': (int(state[0]), int(state[1])),
                                 'value': []}
            for action, action_value in state_value.items():
                action_transitions = {'action': int(action),
                                      'value': []}
                for next_state, next_state_value in action_value.items():
                    next_state_transitions = {'next_state': (int(next_state[0]), int(next_state[1])),
                                              'value': float(next_state_value)}
                    action_transitions['value'].append(next_state_transitions)
                state_transitions['value'].append(action_transitions)
            transitions.append(state_transitions)
        json_dict['P'] = transitions
        return json_dict

    @classmethod
    def from_json_dict(cls, json_dict):
        json_dict = json_dict.copy()
        transition_model = cls(json_dict['actions'])
        transitions = json_dict['P']
        for state_value in transitions:
            for action_value in state_value['value']:
                for next_state_value in action_value['value']:
                    transition_model.set_transition(state_value['state'], action_value['action'],
                                                    next_state_value['next_state'], next_state_value['value'])
        return transition_model


class RewardFunction:

    def __init__(self, actions, default_reward=0):
        self.actions = actions
        self.R = defaultdict(lambda: defaultdict(lambda: default_reward))

    def set_reward(self, state, action, reward):
        self.R[tuple(state)][action] = reward

    def get_reward(self, state, action=None):
        if action is not None:
            assert action in self.actions
        if action is None:
            return list(self.R[tuple(state)][action_] for action_ in self.actions)
        else:
            return self.R[tuple(state)][action]

    def to_json_dict(self):
        json_dict = self.__dict__.copy()
        rewards = []
        for state, state_value in self.R.items():
            state_rewards = {'state': (int(state[0]), int(state[1])),
                             'value': []}
            for action, action_value in state_value.items():
                action_rewards = {'action': int(action),
                                  'value': float(action_value)}
                state_rewards['value'].append(action_rewards)
            rewards.append(state_rewards)
        json_dict['R'] = rewards
        return json_dict

    @classmethod
    def from_json_dict(cls, json_dict):
        json_dict = json_dict.copy()
        reward_function = cls(json_dict['actions'])
        rewards = json_dict['R']
        for state_value in rewards:
            for action_value in state_value['value']:
                reward_function.set_reward(state_value['state'], action_value['action'], action_value['value'])
        return reward_function
