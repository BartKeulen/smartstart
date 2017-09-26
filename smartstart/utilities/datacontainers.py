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

    def __init__(self):
        super().__init__()
        self.obs = []
        self.action = []
        self.reward = []
        self.obs_tp1 = []
        self.done = []

    def total_reward(self):
        """Total episode reward

        Returns
        -------
        :obj:`float`
            total reward
        """
        return sum(self.reward)

    def average_reward(self):
        """Average episode reward

        Returns
        -------
        :obj:`float`
            average reward
        """
        return sum(self.reward) / len(self.reward)

    def append(self, obs, action, reward, obs_tp1, done):
        """Add transition to episode

        Parameters
        ----------
        obs : :obj:`np.ndarray`
            
        action : :obj:`int`
            
        reward : :obj:`float`

        obs_tp1 : :obj:`np.ndarray`
            
        done : :obj:`bool`
        """
        self.obs.append(obs)
        self.action.append(action)
        self.reward.append(reward)
        self.obs_tp1.append(obs_tp1)
        self.done.append(done)

    def to_json(self):
        """Convert episode data to json string

        Returns
        -------
        :obj:`str`
            JSON string of episode data
        """
        json_dict = {
            'obs': np.asarray(self.obs).tolist(),
            'action': np.asarray(self.action).tolist(),
            'reward': np.asarray(self.reward).tolist(),
            'obs_tp1': np.asarray(self.obs_tp1).tolist(),
            'done': np.asarray(self.done).tolist()
        }
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
        episode = cls()
        episode.__dict__.update(json.loads(data))
        return episode

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, key):
        return self.obs[key], self.action[key], self.reward[key], self.obs_tp1[key], self.done[key]

    def __repr__(self):
        return "%s(length=%d, total_reward=%.2f, average_reward=%.2f)" % \
               (self.__class__.__name__, self.__len__(), self.total_reward(), self.average_reward())


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

    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.episodes = []

    def append(self, episode):
        """Adds the length and total reward of episode to summary

        Parameters
        ----------
        episode : :obj:`~smartstart.utilities.datacontainers.Episode`
            episode object

        """
        self.episodes.append((len(episode), episode.total_reward()))

    def total_reward(self):
        """Total reward of all episodes

        Returns
        -------
        :obj:`float`
            total reward
        """
        return sum(self.total_episode_reward())

    def average_reward(self):
        """Average reward of all episodes

        Returns
        -------
        :obj:`float`
            average reward
        """
        return self.total_reward() / self.__len__()

    def total_episode_reward(self):
        """Total reward per episode

        Returns
        -------
        :obj:`list` of :obj:`float`
            total reward per episode
        """
        return [reward for steps, reward in self.episodes]

    def average_episode_reward(self):
        """Average reward per episode

        Returns
        -------
        :obj:`list` of :obj:`float`
            average reward per episode
        """
        return [reward / steps for steps, reward in self.episodes]

    def steps_episode(self):
        """Number of steps per episode

        Returns
        -------
        :obj:`list` of :obj:`int`
            steps per episode
        """
        return [steps for steps, _ in self.episodes]

    def to_json(self):
        """Convert summary data to JSON string

        Returns
        -------
        :obj:`str`
            JSON string of summary data
        """
        return json.dumps(self.__dict__)

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
        summary = cls()
        data_dict = json.loads(data)
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
        name = self.name
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
        name = self.name
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

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, key):
        return self.episodes[key]

    def __repr__(self):
        return "%s(num_episodes=%d, average_reward=%.2f)" % \
               (self.__class__.__name__, self.__len__(), self.average_reward())
