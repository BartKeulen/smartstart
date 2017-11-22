import csv
import datetime
import os
import random
from multiprocessing import Pool, cpu_count, Manager, Process

import numpy as np
from sklearn.model_selection import ParameterGrid
from google.cloud import storage

from smartstart.environments import GridWorld
from smartstart.algorithms.tdtabular import TDTabular


class SimpleAgent(TDTabular):
    RANDOM = 'random'
    COUNT = 'count-based'

    def __init__(self, *args, **kwargs):
        super(SimpleAgent, self).__init__(*args, **kwargs)

    def train(self, test_freq=0, render=False, render_episode=False, print_results=True):
        total_steps = 0
        for i_episode in range(self.num_episodes):
            obs = self.env.reset()

            for _ in range(self.max_steps):
                action = self.get_action(obs)

                obs_tp1, reward, done = self.env.step(action)

                self.increment(obs, action, obs_tp1)

                obs = obs_tp1

                if render:
                    self.env.render(density_map=self.get_density_map())

                total_steps += 1
                if done:
                    return total_steps

        return np.nan

    def get_next_q_action(self, obs_tp1, done):
        return 0, self.get_action(obs_tp1)

    def get_action(self, obs):
        if self.exploration == self.RANDOM:
            return random.randint(0, 3)
        elif self.exploration == self.COUNT:
            max_value = -float('inf')
            max_actions = []
            for action in range(4):
                value = float('inf')
                count = self.get_count(obs, action)
                if count > 0:
                    obs_tp1 = self.next_obses(obs, action)[0]
                    count_obs_tp1 = self.get_count(obs_tp1)
                    if count_obs_tp1 != 0:
                        value = 1/self.get_count(obs_tp1)
                if value > max_value:
                    max_value = value
                    max_actions = [action]
                elif value == max_value:
                    max_actions.append(action)

            return random.choice(max_actions)


def run_test(args):
    params, queue = args
    print('Process %d started' % params['id'])

    env = GridWorld.generate(params['env'])

    steps = []
    for i in range(params['num_iter']):
        agent = SimpleAgent(env,
                            num_episodes=params['num_episodes'],
                            max_steps=params['max_steps'],
                            exploration=params['exploration_strategy'])

        total_steps = agent.train()
        steps.append(total_steps)

    params['steps'] = steps
    steps = np.asarray(steps)
    params['mean'] = np.mean(steps)
    params['std'] = np.std(steps)

    queue.put(params)


def writer_to_file(queue, filename, fieldnames):
    with open(filename, 'w', newline='\n') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter=';')

        print("Writer opened successful")

        writer.writeheader()
        while True:
            param_set = queue.get()
            if type(param_set) == dict:
                print("Writing process %d" % param_set['id'])
                del param_set['id']
                writer.writerow(param_set)

            if param_set == 'DONE':
                break


def write_to_gcloud(local_fp, bucket, directory):
    name = datetime.datetime.now().strftime('%d%m%Y-%H:%M') + '.csv'
    fp = os.path.join(directory, name)

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket)
    blob = bucket.blob(fp)

    blob.upload_from_filename(local_fp)


def main(n_processes=None, fp=None, save_to_cloud=False, bucket=None, directory=None):
    if fp is None:
        fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

    params = {
        'num_iter': [10],
        'num_episodes': [10000],
        'max_steps': [50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000],
        'env': [GridWorld.EASY, GridWorld.MEDIUM, GridWorld.HARD, GridWorld.EXTREME],
        'exploration_strategy': [SimpleAgent.RANDOM, SimpleAgent.COUNT]
    }
    param_grid = list(ParameterGrid(params))
    for i, param_set in enumerate(param_grid):
        param_set['id'] = i
    fieldnames = ['env', 'exploration_strategy', 'num_iter', 'num_episodes', 'max_steps', 'mean', 'std', 'steps']
    local_fp = os.path.join(fp, 'exploration.csv')

    m = Manager()
    queue = m.Queue()
    writer_p = Process(target=writer_to_file, args=(queue, local_fp, fieldnames))
    writer_p.daemon = True
    writer_p.start()

    if n_processes is None:
        n_processes = cpu_count()
    p = Pool(n_processes - 1)
    p.map(run_test, [(param_set, queue) for param_set in param_grid])
    queue.put('DONE')
    p.close()
    p.join()
    writer_p.join()

    if save_to_cloud:
        write_to_gcloud(local_fp, bucket, directory)


if __name__ == '__main__':
    main(save_to_cloud=True, bucket='smartstart', directory='complexity')
