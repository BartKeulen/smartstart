"""Datacontainers module

Defines datacontainer classes training data.
"""
import json
import os

import numpy as np
from google.cloud import storage

from smartstart.utilities.utilities import DIR


class Episode(object):
    """Datacontainer for episode data

    Attributes
    ----------
    obs : :obj:`list` of :obj:`np.ndarray`
        observations
    action : :obj:`list` of :obj:`int`
        actions
    reward : :obj:`list` of :obj:`float`
        rewards
    obs_tp1 : :obj:`list` of :obj:`np.ndarray`
        next observations
    done : :obj:`list` of :obj:`bool`
        dones
    """

    def __init__(self, iter):
        super().__init__()
        self.iter = iter
        self.steps = 0
        self.reward = 0
        self.value_function = None

    def append(self, reward):
        """Add transition to episode

        Parameters
        ----------
        obs : :obj:`np.ndarray`
            
        action : :obj:`int`
            
        reward : :obj:`float`

        obs_tp1 : :obj:`np.ndarray`
            
        done : :obj:`bool`
        """
        self.steps += 1
        self.reward += reward

    def set_value_function(self, value_function):
        self.value_function = value_function.copy()

    def average_reward(self):
        """Average episode reward

        Returns
        -------
        :obj:`float`
            average reward
        """
        if self.steps == 0:
            return 0.
        return self.reward / self.steps

    def to_json(self):
        """Convert episode data to json string

        Returns
        -------
        :obj:`str`
            JSON string of episode data
        """
        json_dict = self.__dict__
        json_dict['value_function'] = np.asarray(self.value_function).tolist()

        return json.dumps(json_dict)

    @classmethod
    def from_json(cls, data):
        """Coverts JSON string into
        :class:`~smartstart.utilities.datacontainers.Episode` object

        Parameters
        ----------
        data : :obj:`str`
            JSON string

        Returns
        -------
        :obj:`~smartstart.utilities.datacontainers.Episode`
            new episode object
        """
        json_dict = json.loads(data)
        episode = cls(json_dict['iter'])
        json_dict['value_function'] = np.asarray(json_dict['value_function'])
        episode.__dict__.update(json_dict)
        return episode

    def __len__(self):
        return self.steps

    def __repr__(self):
        return "%s(steps=%d, total_reward=%.2f)" % \
               (self.__class__.__name__, self.__len__(), self.reward)


class Summary(object):
    """Datacontainer for complete training session

    Parameters
    ----------
    name : :obj:`str`
        name of the summary, used for saving the data (Default = None)

    Attributes
    ----------
    name : :obj:`str`
        name of the summary, used for saving the data (Default = None)
    episodes : :obj:`list` of :obj:`tuple` with episode data
    """

    def __init__(self, name, env_name, max_steps):
        super().__init__()
        self.name = name
        self.env_name = env_name
        self.max_steps = max_steps
        self.episodes = []
        self.tests = []

    def append(self, episode):
        """Adds the length and total reward of episode to summary

        Parameters
        ----------
        episode : :obj:`~smartstart.utilities.datacontainers.Episode`
            episode object

        """
        self.episodes.append(episode)

    def append_test(self, episode):
        self.tests.append(episode)

    def iterations(self, train=True):
        if train:
            episodes = self.episodes
        else:
            episodes = self.tests
        return [episode.iter for episode in episodes]

    def total_reward(self, train=True):
        """Total reward of all episodes

        Returns
        -------
        :obj:`float`
            total reward
        """
        if train:
            episodes = self.episodes
        else:
            episodes = self.tests
        return sum([episode.reward for episode in episodes])

    def average_reward(self, train=True):
        if train:
            episodes = self.episodes
        else:
            episodes = self.tests
        if len(episodes) == 0:
            return 0
        return self.total_reward(train) / len(episodes)

    def steps_episode(self, train=True):
        """Number of steps per episode

        Returns
        -------
        :obj:`list` of :obj:`int`
            steps per episode
        """
        if train:
            episodes = self.episodes
        else:
            episodes = self.tests
        return [episode.steps for episode in episodes]

    def to_json(self):
        """Convert summary data to JSON string

        Returns
        -------
        :obj:`str`
            JSON string of summary data
        """
        json_dict = self.__dict__.copy()
        update_dict = {
            'episodes': [episode.to_json() for episode in self.episodes],
            'tests': [test.to_json() for test in self.tests]
        }
        json_dict.update(update_dict)
        return json.dumps(json_dict)

    @classmethod
    def from_json(cls, data):
        """Convert JSON string into
        :class:`~smartstart.utilities.datacontainers.Summary` object

        Parameters
        ----------
        data : :obj:`str`
            JSON string with summary data

        Returns
        -------
        :obj:`~smartstart.utilities.datacontainers.Summary`
            new Summary object
        """
        summary = cls(None, None, None)
        data_dict = json.loads(data)
        data_dict['episodes'] = [Episode.from_json(episode) for episode in data_dict['episodes']]
        data_dict['tests'] = [Episode.from_json(test) for test in data_dict['tests']]
        summary.__dict__.update(data_dict)
        return summary

    def save(self, directory=DIR, post_fix=None):
        """Save summary as json file

        The summary name is used as filename, an optional postfix can be
        added to the end of the summary name.

        Parameters
        ----------
        directory : :obj:`str`
             directory to save the summary (Default value = DIR)
        post_fix : :obj:`str`
             post_fix to add to the end of the summary name (Default value =
             None)

        Returns
        -------
        :obj:`str`
            full filepath to the saved json summary
        """
        name = self.name + "_" + self.env_name
        if post_fix is not None:
            name += "_" + str(post_fix)
        name += ".json"

        fp = os.path.join(directory, name)

        with open(fp, 'w') as f:
            f.write(self.to_json())

        return fp

    def save_to_gcloud(self, bucket_name, directory, post_fix=None):
        """Save summary in a Google Cloud Bucket

        Note:
            Google cloud SDK has to be installed and initialized before
            function can be used. Click `here`_ for more information.

        .. _here: https://cloud.google.com/compute/docs/tutorials/python-guide

        Parameters
        ----------
        directory : :obj:`str`
             directory to save the summary (Default value = DIR)
        post_fix : :obj:`str`
             post_fix to add to the end of the summary name (Default value =
             None)

        """
        name = self.name + "_" + self.env_name
        if post_fix is not None:
            name += "_" + str(post_fix)
        name += ".json"
        fp = os.path.join(directory, name)

        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(fp)

        blob.upload_from_string(self.to_json())

    @classmethod
    def load(cls, fp):
        """Loads a summary from the defined filepath

        Parameters
        ----------
        fp : :obj:`str`
            filepath to the summary

        Returns
        -------
        :obj:`~smartstart.utilities.datacontainers.Summary`
            new summary object
        """
        with open(fp, 'r') as f:
            data = f.read()
            return cls.from_json(data)
