import numpy as np
import pdb
import random


def greedy(state, state_action_values):
    values_ = state_action_values[state[0], state[1], :]
    return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])


def epsilon_greedy(state, state_action_values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(state_action_values[state[0], state[1], :].shape[0])
    else:
        values_ = state_action_values[state[0], state[1], :]
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])


def boltzmann(state, state_action_values, temp):
    values_ = np.exp(state_action_values[state[0], state[1], :] / temp)
    return np.random.choice(np.arange(values_.shape[0]), p=values_ / np.sum(values_))


def ucb1(state, state_action_values, visitation_counts, c):
    values_ = state_action_values[state[0], state[1], :]
    counts_ = visitation_counts.get_state_action_visitation_counts(state)
    # counts_ = visitation_counts[state[0], state[1], :]

    if np.sum(counts_) == 0:
        np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

    values = []
    for value, count in zip(values_, counts_):
        if count == 0:
            value = np.inf
        else:
            bonus = c * np.sqrt(np.log(np.sum(counts_)) / count)
            value += bonus
        values.append(value)
    return np.random.choice([action_ for action_, value_ in enumerate(values) if value_ == np.max(values)])
