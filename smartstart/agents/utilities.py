import pdb
import os
import json
import logging

import numpy as np
from google.cloud import storage

from smartstart.agents.qlearning import QLearning
from smartstart.agents.rmax import RMax
from smartstart.agents.mbrl import MBRL
from smartstart.agents.smartstart import SmartStart
from smartstart.environments.gridworld import GridWorld

logger = logging.getLogger(__name__)

RENDER = False
RENDER_TEST = False
RENDER_EPISODE = False

SEED = None
MAX_STEPS = 50000
MAX_STEPS_EPISODE = 100
TEST_FREQ = 0

agents = {
    'QLearning': QLearning,
    'RMax': RMax,
    'MBRL': MBRL,
    'SmartStart': SmartStart
}


class Episode:

    def __init__(self, iter):
        self.iter = iter
        self.steps = 0
        self.reward = 0.

    def add(self, reward):
        self.steps += 1
        self.reward += reward


class Summary:

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.seed = SEED
        self.max_steps = MAX_STEPS
        self.max_steps_episode = MAX_STEPS_EPISODE
        self.test_freq = TEST_FREQ
        self.train = {
            'iter': [],
            'steps': [],
            'rewards': []
        }
        self.test = {
            'iter': [],
            'steps': [],
            'rewards': []
        }
        self.correct_policy = {
            'iter': [],
            'correct_policy': []
        }

    def add_train_episode(self, episode):
        self._add_episode(self.train, episode)

    def add_test_episode(self, episode):
        self._add_episode(self.test, episode)

    def add_test_policy(self, iter, percentage_correct):
        self.correct_policy['iter'].append(iter)
        self.correct_policy['correct_policy'].append(percentage_correct)

    def _add_episode(self, episode_list, episode):
        episode_list['iter'].append(episode.iter)
        episode_list['steps'].append(episode.steps)
        episode_list['rewards'].append(episode.reward)

    def get_train_iterations_in_training_steps(self):
        return np.cumsum(self.train['steps'])

    def get_test_iterations_in_training_steps(self):
        training_steps = self.get_train_iterations_in_training_steps()
        return np.asarray([training_steps[i] for i in self.test['iter']])

    def get_policy_iterations_in_training_steps(self):
        training_steps = self.get_train_iterations_in_training_steps()
        return np.asarray([training_steps[i] for i in self.correct_policy['iter']])

    def to_json_dict(self):
        json_dict = self.__dict__.copy()
        json_dict['env'] = self.env.to_json_dict()
        json_dict['agent'] = self.agent.to_json_dict()
        return json_dict

    def to_json(self):
        return json.dumps(self.to_json_dict())

    @classmethod
    def from_json(cls, json_str):
        json_dict = json.loads(json_str)
        json_dict = json_dict.copy()
        env = GridWorld.from_json_dict(json_dict['env'])
        agent = agents[json_dict['agent']['name']].from_json_dict(json_dict['agent'])
        del json_dict['env']
        del json_dict['agent']
        summary = cls(env, agent)
        summary.__dict__.update(json_dict)
        return summary

    def save(self, directory, post_fix=None, bucket_name=None):
        agent = self.agent
        name = agent.name
        if agent.__class__ == SmartStart:
            agent = self.agent.agent
            name += '_' + agent.name
        if hasattr(agent, 'exploration_strategy'):
            name += '_%s' % agent.exploration_strategy
        name += '_%s' % self.env.name
        if post_fix is not None:
            name += '_%s' % post_fix
        name += '.json'
        fp = os.path.join(directory, name)

        return self.save_fp(fp, bucket_name)

    def save_fp(self, fp, bucket_name=None):
        if bucket_name is None:
            create_directory(fp)
            with open(fp, 'w') as fw:
                fw.write(self.to_json())
        else:
            storage_client = storage.Client()
            bucket = storage_client.get_bucket(bucket_name)
            blob = bucket.blob(fp)
            blob.upload_from_string(self.to_json())
        return fp

    @classmethod
    def load(cls, fp):
        with open(fp, 'r') as fr:
            data = fr.read()
            return cls.from_json(data)


def create_directory(fp):
    """Creates and returns a data directory at the file's location

    Parameters
    ----------
    fp :
        python file

    Returns
    -------
    :obj:`str`
        filepath to the data directory

    """
    fp = os.path.abspath(os.path.dirname(fp))
    if not os.path.exists(fp):
        os.makedirs(fp)
    return fp


def calc_average_reward_training_steps(summaries):
    average_rewards = []
    training_steps = []
    for summary in summaries:
        average_rewards.append(np.asarray(summary.test['rewards']) / np.asarray(summary.test['steps']))
        training_steps.append(summary.get_test_iterations_in_training_steps())
    average_rewards = np.asarray(average_rewards)
    training_steps = np.asarray(training_steps)

    x = np.unique(np.concatenate(training_steps))
    y = np.asarray([np.interp(x, steps, rewards) for steps, rewards in zip(training_steps, average_rewards)])
    mean_y = np.mean(y, axis=0)
    std = np.std(y, axis=0)

    return x, mean_y, std


def calc_average_steps_training_steps(summaries):
    average_steps = []
    training_steps = []
    for summary in summaries:
        average_steps.append(np.asarray(summary.test['steps']))
        training_steps.append(summary.get_test_iterations_in_training_steps())
    average_steps = np.asarray(average_steps)
    training_steps = np.asarray(training_steps)

    x = np.unique(np.concatenate(training_steps))
    y = np.asarray([np.interp(x, training_step, step) for training_step, step in zip(training_steps, average_steps)])
    mean_y = np.mean(y, axis=0)
    std = np.std(y, axis=0)

    return x, mean_y, std


def calc_average_rise_time_in_training_steps(summaries, epsilon, baseline):
    rise_times = []
    for summary in summaries:
        idx = np.argmax(np.asarray(summary.test['steps']) <= np.ceil(baseline * (1 + epsilon)))
        training_steps = summary.get_test_iterations_in_training_steps()
        rise_time = training_steps[idx] if idx > 0 else training_steps[-1]
        rise_times.append(rise_time)
    rise_times = np.asarray(rise_times)

    mean = np.mean(rise_times)
    std = np.std(rise_times)
    return mean, std

# def calc_average_rise_time_in_training_steps(summaries, epsilon, baseline):
#     training_steps, average_steps, std_steps = calc_average_steps_training_steps(summaries)
#
#     lower_steps = average_steps - std_steps
#     upper_steps = average_steps + std_steps
#
#     idx = np.argmax(average_steps <= (baseline * (1 + epsilon)))
#     lower_idx = np.argmax(lower_steps <= (baseline * (1 + epsilon)))
#     upper_idx = np.argmax(upper_steps <= (baseline * (1 + epsilon)))
#
#     rise_time = training_steps[idx] if idx > 0 else np.nan
#
#     rise_time_lower = training_steps[lower_idx] if lower_idx > 0 else np.nan
#     rise_time_upper = training_steps[upper_idx] if upper_idx > 0 else np.nan
#     return rise_time, rise_time_lower, rise_time_upper


def calc_average_policy_in_training_steps(summaries):
    average_policies = []
    training_steps = []
    for summary in summaries:
        average_policies.append(np.asarray(summary.correct_policy['correct_policy']))
        training_steps.append(summary.get_policy_iterations_in_training_steps())
    average_policies = np.asarray(average_policies)
    training_steps = np.asarray(training_steps)

    x = np.unique(np.concatenate(training_steps))
    y = np.asarray([np.interp(x, training_step, policy) for training_step, policy in zip(training_steps, average_policies)])
    mean_y = np.mean(y, axis=0)
    std = np.std(y, axis=0)

    return x, mean_y, std
