import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
from pathlib import Path

import numpy as np

from smartstart.utilities.datacontainers import Episode, Summary


def random_episode(i_episode, length):
    episode = Episode(i_episode)
    tot_reward = 0
    for i in range(length):
        reward = np.random.randn()
        episode.append(reward)
        tot_reward += reward
    return episode, tot_reward


def random_summary(name, length, ep_length, v_size):
    rewards = []
    value_functions = []
    summary = Summary(name)
    for i in range(length):
        episode, reward = random_episode(i, ep_length)
        value_function = np.random.randn(v_size, v_size)
        episode.set_value_function(value_function)
        summary.append(episode)
        rewards.append(reward)
        value_functions.append(value_function)
    return summary, rewards, value_functions


class TestEpisode:
    iter = 0
    length = 1000

    def test_append(self):
        episode, tot_reward = random_episode(self.iter, self.length)

        assert len(episode) == self.length
        assert episode.reward == tot_reward

    def test_set_value_function(self):
        value_function = np.random.randn(5, 5)
        episode, tot_reward = random_episode(self.iter, self.length)
        episode.set_value_function(value_function)

        assert episode.value_function is not value_function
        np.testing.assert_array_equal(episode.value_function, value_function)

    def test_average_reward_length_0(self):
        episode = Episode(self.iter)

        assert episode.average_reward() == 0

    def test_average_reward(self):
        episode, tot_reward = random_episode(self.iter, self.length)

        assert episode.average_reward() == tot_reward / self.length

    def test_to_json_without_value_function(self):
        episode, tot_reward = random_episode(self.iter, self.length)

        json_str = episode.to_json()
        json_dict = json.loads(json_str)

        assert json_dict['iter'] == self.iter
        assert json_dict['reward'] == tot_reward
        assert json_dict['steps'] == self.length
        assert json_dict['value_function'] is None

    def test_to_json_with_value_function(self):
        value_function = np.random.randn(5, 5)

        episode, tot_reward = random_episode(self.iter, self.length)
        episode.set_value_function(value_function)

        json_str = episode.to_json()
        json_dict = json.loads(json_str)

        assert json_dict['iter'] == self.iter
        assert json_dict['reward'] == tot_reward
        assert json_dict['steps'] == self.length
        np.testing.assert_array_equal(np.asarray(json_dict['value_function']), value_function)

    def test_from_json(self):
        value_function = np.random.randn(5, 5)

        episode, tot_reward = random_episode(self.iter, self.length)
        episode.set_value_function(value_function)

        json_str = episode.to_json()
        new_episode = Episode.from_json(json_str)

        assert new_episode.iter == episode.iter
        assert new_episode.steps == episode.steps
        assert new_episode.reward == episode.reward
        np.testing.assert_array_equal(new_episode.value_function, episode.value_function)

    def test_length(self):
        episode, tot_reward = random_episode(self.iter, self.length)

        assert len(episode) == self.length


class TestSummary:
    name = 'test'
    length = 1000
    ep_length = 1000
    v_size = 5

    def test_append(self):

        summary, rewards, value_functions = random_summary(self.name, self.length, self.ep_length, self.v_size)

        assert summary.name == self.name
        assert len(summary) == self.length
        for episode, reward, value_function in zip(summary.episodes, rewards, value_functions):
            assert episode.reward == reward
            assert episode.steps == self.ep_length
            np.testing.assert_array_equal(episode.value_function, value_function)

    def test_total_reward(self):
        summary, rewards, _ = random_summary(self.name, self.length, self.ep_length, self.v_size)

        assert summary.total_reward() == sum(rewards)

    def test_average_reward_length_0(self):
        summary = Summary(self.name)

        assert summary.average_reward() == 0

    def test_average_reward(self):
        summary, rewards, _ = random_summary(self.name, self.length, self.ep_length, self.v_size)

        assert summary.average_reward() == sum(rewards) / self.length

    def test_steps_episode(self):
        summary, rewards, _ = random_summary(self.name, self.length, self.ep_length, self.v_size)

        for steps in summary.steps_episode():
            assert steps == self.ep_length

    def test_to_json(self):
        summary, _, _ = random_summary(self.name, self.length, self.ep_length, self.v_size)

        json_str = summary.to_json()

        json_dict = json.loads(json_str)
        json_dict['episodes'] = [Episode.from_json(episode) for episode in json_dict['episodes']]
        json_dict['tests'] = [Episode.from_json(test) for test in json_dict['tests']]

        assert json_dict['name'] == summary.name
        for json_episode, episode in zip(json_dict['episodes'], summary.episodes):
            assert json_episode.iter == episode.iter
            assert json_episode.steps == episode.steps
            assert json_episode.reward == episode.reward
            np.testing.assert_array_equal(json_episode.value_function, episode.value_function)

    def test_from_json(self):
        summary, _, _ = random_summary(self.name, self.length, self.ep_length, self.v_size)

        json_str = summary.to_json()
        new_summary = Summary.from_json(json_str)

        assert new_summary.name == summary.name
        for new_episode, episode in zip(new_summary.episodes, summary.episodes):
            assert new_episode.iter == episode.iter
            assert new_episode.steps == episode.steps
            assert new_episode.reward == episode.reward
            np.testing.assert_array_equal(new_episode.value_function, episode.value_function)

    def test_save_without_postfix(self, tmpdir):
        directory = tmpdir.mkdir('data')

        summary, _, _ = random_summary(self.name, self.length, self.ep_length, self.v_size)

        fp = summary.save(directory.strpath)

        assert fp == directory.join(self.name + '.json').strpath
        assert Path(fp).is_file()

    def test_save_with_postfix(self, tmpdir):
        post_fix = "run_0"
        directory = tmpdir.mkdir('data')

        summary, _, _ = random_summary(self.name, self.length, self.ep_length, self.v_size)

        fp = summary.save(directory.strpath, post_fix=post_fix)

        assert fp == directory.join(self.name + '_' + post_fix + '.json').strpath
        assert Path(fp).is_file()

    def test_load(self, tmpdir):
        directory = tmpdir.mkdir('data')

        summary, _, _ = random_summary(self.name, self.length, self.ep_length, self.v_size)

        fp = summary.save(directory.strpath)

        new_summary = Summary.load(fp)

        assert new_summary.name == summary.name
        for new_episode, episode in zip(new_summary.episodes, summary.episodes):
            assert new_episode.iter == episode.iter
            assert new_episode.steps == episode.steps
            assert new_episode.reward == episode.reward
            np.testing.assert_array_equal(new_episode.value_function, episode.value_function)

    def test_length(self):
        summary, _, _ = random_summary(self.name, self.length, self.ep_length, self.v_size)

        assert len(summary) == self.length
