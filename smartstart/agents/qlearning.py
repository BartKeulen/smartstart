import logging
from smartstart.utilities.counter import Counter

from smartstart.utilities.policies import *


logger = logging.getLogger(__name__)


class QLearning:
    GREEDY = 'greedy'
    E_GREEDY = 'e-greedy'
    BOLTZMANN = 'boltzmann'
    UCB1 = 'ucb1'

    def __init__(self,
                 state_action_values,
                 counter,
                 alpha=0.1,
                 gamma=0.99,
                 exploration_strategy=E_GREEDY,
                 epsilon=0.1,
                 temp=10.,
                 c=0.001):
        self.name = self.__class__.__name__
        self.alpha = alpha
        self.gamma = gamma
        self.exploration_strategy = exploration_strategy
        self.epsilon = epsilon
        self.temp = temp
        self.c = c

        self.state_action_values = state_action_values
        self.counter = counter

    def get_greedy_action(self, state):
        return greedy(state, self.state_action_values)

    def get_action(self, state, use_ss_policy=False):
        if self.exploration_strategy == self.GREEDY:
            return greedy(state, self.state_action_values)
        elif self.exploration_strategy == self.E_GREEDY:
            return epsilon_greedy(state, self.state_action_values, self.epsilon)
        elif self.exploration_strategy == self.BOLTZMANN:
            return boltzmann(state, self.state_action_values, self.temp)
        elif self.exploration_strategy == self.UCB1:
            return ucb1(state, self.state_action_values, self.counter, self.c)
        else:
            raise NotImplementedError()

    def update(self, state, action, reward, next_state, done):
        self.counter.increment(state, action, next_state)

        state_action_value = self.state_action_values[state[0], state[1], action]
        if done:
            next_state_action_value = 0
        else:
            next_state_action_value = np.max(self.state_action_values[next_state[0], next_state[1], :])

        td_error = reward + self.gamma * next_state_action_value - state_action_value
        self.state_action_values[state[0], state[1], action] += self.alpha * td_error

    def get_state_values(self):
        return np.max(self.state_action_values, axis=-1)

    def get_state_action_values(self):
        return self.state_action_values

    def to_json_dict(self):
        json_dict = self.__dict__.copy()
        json_dict['state_action_values'] = json_dict['state_action_values'].tolist()
        json_dict['counter'] = self.counter.to_json_dict()
        return json_dict

    @classmethod
    def from_json_dict(cls, json_dict):
        json_dict = json_dict.copy()
        state_action_values = np.asarray(json_dict['state_action_values'])
        counter = Counter.from_json_dict(json_dict['counter'])
        del json_dict['name']
        del json_dict['state_action_values']
        del json_dict['counter']
        agent = cls(state_action_values, counter, **json_dict)
        return agent
