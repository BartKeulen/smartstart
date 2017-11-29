import datetime
import os
import pprint
import random
import argparse

import numpy as np
from multiprocessing import cpu_count

from smartstart.utilities.experimenter import run_experiment
from smartstart.algorithms import QLearning, SARSA, SARSALambda
from smartstart.environments import GridWorld
from smartstart.smartexploration import generate_smartstart_object
from smartstart.utilities.utilities import get_data_directory

directory = get_data_directory(__file__)


def task(params):
    np.random.seed()
    random.seed()

    env = GridWorld.generate(params['env'])

    kwargs = {
        'alpha': 0.1,
        'gamma': 0.99,
        'epsilon': 0.05,
        'steps_episode': params['steps_episode'],
        'max_steps': params['max_steps'],
        'exploration_strategy': params['exploration_strategy']
    }

    if params['use_smart_start']:
        agent = generate_smartstart_object(params['algorithm'],
                                           env,
                                           exploitation_param=0.,
                                           **kwargs)
    else:
        agent = params['algorithm'](env,
                                    **kwargs)

    post_fix = "%0.3d" % params['run']

    summary = agent.train(test_freq=5, render=False, render_episode=False, print_results=False)

    if params['save_to_cloud']:
        summary.save_to_gcloud(bucket_name='smartstart',
                               directory="%s/%s" % (params['directory'], env.name),
                               post_fix=post_fix)
    else:
        summary.save(os.path.join(directory, env.name), post_fix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str, help='GridWorld environment to use; Easy, Medium, Hard, Extreme')
    parser.add_argument('algo', type=str, help='Algorithm to use; QLearning, SARSA, SARSALambda')
    parser.add_argument('num_exp', type=int, help='Number of experiments with each configuration')
    parser.add_argument('-save_to_cloud', action='store_true')
    args = parser.parse_args()

    env = args.env
    if env == GridWorld.EASY:
        steps_episode = 1000
        max_steps = 250000
    elif env == GridWorld.MEDIUM:
        steps_episode = 2500
        max_steps = 5000000
    elif env == GridWorld.HARD:
        steps_episode = 5000
        max_steps = 10000000
    elif env == GridWorld.EXTREME:
        steps_episode = 10000
        max_steps = 50000000
    else:
        raise NotImplementedError("Choose from available environments")

    algo = args.algo
    if algo == 'QLearning':
        algo = QLearning
    elif algo == 'SARSA':
        algo = SARSA
    elif algo == 'SARSALambda':
        algo = SARSALambda
    else:
        raise NotImplementedError("Choose from available algorithms")

    param_grid = {'task': task,
                  'env': [env],
                  'max_steps': [max_steps],
                  'steps_episode': [steps_episode],
                  'algorithm': [algo],
                  'exploration_strategy': [QLearning.E_GREEDY, QLearning.BOLTZMANN, QLearning.COUNT_BASED, QLearning.UCB1],
                  'use_smart_start': [True, False],
                  'num_exp': args.num_exp,
                  'save_to_cloud': [args.save_to_cloud],
                  'directory': [datetime.datetime.now().strftime('%d%m%Y')]}

    pp = pprint.PrettyPrinter()
    pp.pprint(param_grid)

    run_experiment(param_grid)
