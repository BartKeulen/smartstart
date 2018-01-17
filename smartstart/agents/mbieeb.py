import numpy as np
import pdb
from smartstart.agents.valueiteration import ValueIteration


class MBIEEB:

    def __init__(self,
                 counter,
                 vi,
                 m=np.inf,
                 beta=1.):
        self.counter = counter
        self.vi = vi
        self.m = m
        self.beta = beta

        self.state_list = set()
        self.r_sum = np.zeros(counter.shape)

    def get_greedy_action(self, state):
        return self.vi.get_greedy_action(state)

    def get_action(self, state):
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

        if done:
            self.vi.terminal_states.append(tuple(next_state))

        self.vi.state_list = list(self.state_list)
        self.vi.optimize()

    def get_state_values(self):
        return self.vi.state_values


class ValueIterationEB(ValueIteration):

    def __init__(self,
                 state_action_shape,
                 counter,
                 transition_model=None,
                 reward_function=None,
                 terminal_states=None,
                 state_list=None,
                 beta=1.,
                 gamma=0.99,
                 min_error=1e-3,
                 max_itr=1000):
        super(ValueIterationEB, self).__init__(state_action_shape, transition_model, reward_function, terminal_states, state_list,
                                               gamma, min_error, max_itr)
        self.counter = counter
        self.beta = beta

    def get_greedy_action(self, state):
        values = super().calc_state_action_values(state)
        return np.argmax(values)

    def get_action(self, state):
        values = self.calc_state_action_values(state)
        return np.argmax(values)

    def calc_state_action_values(self, state):
        rewards = self.r.get_reward(state)
        ebs = [self.beta / (count + 1) for count in self.counter.get_state_action_visitation_counts(state)]
        values = []
        for transition_probs, next_states, reward, eb in zip(*self.p.get_transition(state), rewards, ebs):
            next_state_values = [0. if next_state in self.terminal_states else
                                 self.state_values[next_state[0], next_state[1]] for next_state in next_states]

            values.append(reward + eb + self.gamma * sum([transition_prob * next_state_value for
                                                          transition_prob, next_state_value in
                                                          zip(transition_probs, next_state_values)]))
        return values