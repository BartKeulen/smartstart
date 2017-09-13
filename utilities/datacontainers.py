import datetime
import json
import os

import numpy as np
from google.cloud import storage

from .utilities import DIR
import gcloud_config

summary_variables = dict()
summary_variables['reward'] = 0


class Episode(object):

    def __init__(self):
        super().__init__()
        self._obs = []
        self._action = []
        self._reward = []
        self._obs_tp1 = []
        self._done = []

    def total_reward(self):
        return sum(self._reward)

    def average_reward(self):
        return sum(self._reward) / len(self._reward)

    def append(self, obs, action, reward, obs_tp1, done):
        self._obs.append(obs)
        self._action.append(action)
        self._reward.append(reward)
        self._obs_tp1.append(obs_tp1)
        self._done.append(done)

    def to_json(self):
        json_dict = {
            '_obs': np.asarray(self._obs).tolist(),
            '_action': np.asarray(self._action).tolist(),
            '_reward': np.asarray(self._reward).tolist(),
            '_obs_tp1': np.asarray(self._obs_tp1).tolist(),
            '_done': np.asarray(self._done).tolist()
        }
        return json.dumps(json_dict)

    @classmethod
    def from_json(cls, data):
        episode = cls()
        episode.__dict__.update(json.loads(data))
        return episode

    def __len__(self):
        return len(self._obs)

    def __getitem__(self, key):
        return self._obs[key], self._action[key], self._reward[key], self._obs_tp1[key], self._done[key]

    def __repr__(self):
        return "%s(length=%d, total_reward=%.2f, average_reward=%.2f)" % \
               (self.__class__.__name__, self.__len__(), self.total_reward(), self.average_reward())


class Summary(object):
    REWARD = 0

    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.time = datetime.datetime.today().isoformat()
        self._episodes = []

    def append(self, episode):
        self._episodes.append((len(episode), episode.total_reward()))

    def total_reward(self):
        return sum(self.total_episode_reward())

    def average_reward(self):
        return self.total_reward() / self.__len__()

    def total_episode_reward(self):
        return [reward for steps, reward in self._episodes]

    def average_episode_reward(self):
        return [reward/steps for steps, reward in self._episodes]

    def to_json(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, data):
        summary = cls()
        data_dict = json.loads(data)
        summary.__dict__.update(data_dict)
        return summary

    def save(self, directory=DIR, post_fix=None):
        name = self.name
        if post_fix is not None:
            name += "_" + str(post_fix)
        name += ".json"

        fp = os.path.join(directory, name)

        with open(fp, 'w') as f:
            f.write(self.to_json())

        return fp

    def save_to_gcloud(self, directory, post_fix=None):
        name = self.name
        if post_fix is not None:
            name += "_" + str(post_fix)
        name += ".json"
        fp = os.path.join(directory, name)

        storage_client = storage.Client()
        bucket = storage_client.get_bucket(gcloud_config.bucket_name)
        blob = bucket.blob(fp)

        blob.upload_from_string(self.to_json())

    @classmethod
    def load(cls, fp):
        with open(fp, 'r') as f:
            data = f.read()
            return cls.from_json(data)

    def __len__(self):
        return len(self._episodes)

    def __getitem__(self, key):
        return self._episodes[key]

    def __repr__(self):
        return "%s(num_episodes=%d, average_reward=%.2f)" % \
               (self.__class__.__name__, self.__len__(), self.average_reward())
