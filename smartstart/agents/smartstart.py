import numpy as np
from smartstart.agents.counter import Counter
from smartstart.agents.qlearning import QLearning
from smartstart.agents.rmax import RMax
from smartstart.agents.mbrl import MBRL
from smartstart.agents.valueiteration import TransitionModel, RewardFunction, ValueIteration

agents = {
    'QLearning': QLearning,
    'RMax': RMax,
    'MBRL': MBRL
}


class SmartStart:

    def __init__(self,
                 agent,
                 value_iteration,
                 counter,
                 c_ss=2.,
                 eta=0.5,
                 m=1.):
        self.name = self.__class__.__name__
        self.agent = agent
        self.vi = value_iteration
        self.counter = counter

        self.c_ss = c_ss
        self.eta = eta
        self.m = m

    def get_greedy_action(self, state):
        return self.agent.get_greedy_action(state)

    def get_action(self, state, use_ss_policy=False):
        if use_ss_policy:
            return self.vi.get_action(state)
        else:
            return self.agent.get_action(state)

    def get_smart_start_state(self):
        state_visitation_counts = self.counter.get_state_visitation_counts()
        total_count = np.sum(state_visitation_counts)
        state_values = self.agent.get_state_values()

        with np.errstate(divide='ignore'):
            ucb = state_values + np.sqrt((self.c_ss * np.log(total_count)) / state_visitation_counts)
        ucb[ucb == np.inf] = 0

        smart_start_states = np.where(np.ravel(ucb) == np.max(ucb))[0]  # Returns flattened indices of values that equal maximum ucb
        return np.unravel_index(np.random.choice(smart_start_states), ucb.shape)  # Chooses random and transforms back to state

    def fit_model_and_optimize(self, smart_start_state):
        actions = list(range(self.counter.shape[-1]))
        transition_model = TransitionModel(actions)
        reward_function = RewardFunction(actions)

        visited_states = self.counter.get_visited_states()
        for state in visited_states:
            for action in actions:
                visitation_counts, next_states = self.counter.get_state_action_state_visitation_counts(state, action)
                for visitation_count, next_state in zip(visitation_counts, next_states):
                    transition_prob = visitation_count / np.sum(visitation_counts)
                    transition_model.set_transition(state, action, next_state, transition_prob)

                    if next_state == smart_start_state:
                        reward_function.set_reward(state, action, 1.)

        transition_model.validate()

        self.vi.set(transition_model, reward_function, [smart_start_state])

        self.vi.optimize()

    def update(self, obs, action, reward, obs_tp1, done):
        self.agent.update(obs, action, reward, obs_tp1, done)

    def get_state_values(self):
        return self.agent.get_state_values()

    def get_state_action_values(self):
        return self.agent.get_state_action_values()

    def to_json_dict(self):
        json_dict = self.__dict__.copy()
        json_dict['agent'] = self.agent.to_json_dict()
        json_dict['counter'] = self.counter.to_json_dict()
        json_dict['vi'] = self.vi.to_json_dict()
        return json_dict

    @classmethod
    def from_json_dict(cls, json_dict):
        json_dict = json_dict.copy()
        agent = agents[json_dict['agent']['name']].from_json_dict(json_dict['agent'])
        counter = Counter.from_json_dict(json_dict['counter'])
        vi = ValueIteration.from_json_dict(json_dict['vi'])
        del json_dict['agent']
        del json_dict['counter']
        del json_dict['vi']
        del json_dict['name']
        agent = cls(agent, vi, counter, **json_dict)
        return agent


