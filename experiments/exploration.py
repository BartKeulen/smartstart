import csv
import datetime
import os
import random
from multiprocessing import Pool, cpu_count, Manager, Process

import numpy as np
from sklearn.model_selection import ParameterGrid
from google.cloud import storage

from smartstart.algorithms.dynamicprogramming import ValueIteration
from smartstart.environments import GridWorld
from smartstart.algorithms.tdtabular import TDTabular
from smartstart.environments.gridworldvisualizer import GridWorldVisualizer


class SimpleAgent(TDTabular):
    RANDOM = 'random'
    COUNT = 'count-based'

    def __init__(self, env, smart_start=False, *args, **kwargs):
        super(SimpleAgent, self).__init__(env, *args, **kwargs)
        self.smart_start = smart_start
        self.vi = ValueIteration(self.env)

    def train(self, test_freq=0, render=False, render_episode=False, print_results=True):
        total_steps = 0
        for i_episode in range(self.num_episodes):
            obs = self.env.reset()

            remaining_steps = self.max_steps
            if self.smart_start:
                smart_start_state = self.get_smart_start_state()

                if smart_start_state is not None:
                    self.dynamic_programming(smart_start_state)

                    for i in range(self.max_steps):
                        action = self.vi.get_action(obs)

                        obs_tp1, reward, done = self.env.step(action)

                        self.increment(obs, action, obs_tp1)

                        obs = obs_tp1

                        if render:
                            self.env.render(density_map=self.get_density_map())

                        total_steps += 1

                        if done:
                            return total_steps

                        if np.array_equal(obs, smart_start_state):
                            remaining_steps -= i
                            break

            for _ in range(remaining_steps):
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

    def get_smart_start_state(self):
        count_map = self.get_count_map()
        possible_starts = np.asarray(np.where(count_map > 0))
        if not possible_starts.any():
            return None

        smart_start_state = None
        max_value = -float('inf')
        for i in range(possible_starts.shape[1]):
            obs = possible_starts[:, i]
            value = 1/count_map[tuple(obs)]
            if value > max_value:
                smart_start_state = obs
                max_value = value
        return smart_start_state

    def dynamic_programming(self, smart_start_state):
        self.vi.reset()

        count_map = self.get_count_map()
        states = np.asarray(np.where(count_map > 0))
        obses = []

        for i in range(states.shape[1]):
            obs = states[:, i]
            obses.append(tuple(obs))
            for action in self.env.possible_actions(obs):
                sa_count = self.get_count(obs, action)
                if sa_count >= 1:
                    for next_obs in self.next_obses(obs, action):
                        transition = self.get_count(obs, action, next_obs) / sa_count
                        self.vi.set_transition(obs, action, next_obs, transition)

                        reward = 0.
                        if next_obs == tuple(smart_start_state):
                            reward = 1.
                        self.vi.set_reward(obs, action, reward)

        self.vi.obses = obses

        self.vi.optimize()

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
                            smart_start=params['smart_start'],
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
    name = 'exploration_%s.csv' % datetime.datetime.now().strftime('%d%m%Y-%H:%M')
    fp = os.path.join(directory, name)

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket)
    blob = bucket.blob(fp)

    blob.upload_from_filename(local_fp)


def main(n_processes=None, fp=None, save_to_cloud=False, bucket=None, directory=None):
    if fp is None:
        fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

    params = {
        'num_iter': [5],
        'num_episodes': [1000],
        'max_steps': [50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000],
        'env': [GridWorld.EASY, GridWorld.MEDIUM, GridWorld.HARD, GridWorld.EXTREME],
        'exploration_strategy': [SimpleAgent.RANDOM, SimpleAgent.COUNT],
        'smart_start': [True, False]
    }
    param_grid = list(ParameterGrid(params))
    id = 0
    new_param_grid = []
    for i, param_set in enumerate(param_grid):
        if param_set['env'] == GridWorld.MEDIUM and param_set['max_steps'] < 100:
            continue
        elif param_set['env'] == GridWorld.HARD and param_set['max_steps'] < 250:
            continue
        elif param_set['env'] == GridWorld.EXTREME and param_set['max_steps'] < 250:
            continue
        else:
            param_set['id'] = id
            id += 1
            new_param_grid.append(param_set)
    param_grid = new_param_grid

    fieldnames = ['env', 'exploration_strategy', 'smart_start', 'num_iter', 'num_episodes', 'max_steps', 'mean', 'std', 'steps']
    local_fp = os.path.join(fp, 'exploration_%s.csv' % (datetime.datetime.now().strftime('%d%m%Y-%H:%M')))

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

