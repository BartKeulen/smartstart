import datetime
import pprint
import random
import argparse

import numpy as np

from smartstart.utilities.experimenter import run_experiment
from smartstart.algorithms import QLearning
from smartstart.algorithms import SARSA, SARSALambda
from smartstart.algorithms import TDLearning
from smartstart.environments.gridworld import GridWorld
from smartstart.smartexploration.smartexploration import generate_smartstart_object
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
        'num_episodes': 1000,
        'max_steps': params['max_steps'],
        'exploration': params['exploration_strategy']
    }

    if params['use_smart_start']:
        agent = generate_smartstart_object(params['algorithm'],
                                           env,
                                           exploitation_param=0.,
                                           **kwargs)
    else:
        agent = params['algorithm'](env,
                                    **kwargs)

    post_fix = "exploration=%s_%0.3d" % (params['exploration_strategy'], params['run'])

    summary = agent.train(test_freq=5, render=False, render_episode=False, print_results=False)

    summary.save_to_gcloud(bucket_name='smartstart',
                           directory="%s/%s" % (params['directory'], env.name),
                           post_fix=post_fix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('environment', type=int, help='An integer for the GridWorld environment (0 = Easy, 1 = Medium, 2 = Hard, 3 = Extreme)')
    args = parser.parse_args()

    env = args.environment
    if env == 0:
        max_steps = 1000
    elif env == 1:
        max_steps = 2500
    elif env == 2:
        max_steps = 5000
    elif env == 3:
        max_steps = 10000
    else:
        raise NotImplementedError("Choose from available environments (0 = Easy, 1 = Medium, 2 = Hard, 3 = Extreme)")

    param_grid = {'task': task,
                  'env': [env],
                  'max_steps': [max_steps],
                  'algorithm': [QLearning, SARSA, SARSALambda],
                  'exploration_strategy': [TDLearning.E_GREEDY, TDLearning.BOLTZMANN, TDLearning.COUNT_BASED, TDLearning.UCT],
                  'use_smart_start': [True, False],
                  'num_exp': 25,
                  'directory': [datetime.datetime.now().strftime('%d%m%Y')]}

    pp = pprint.PrettyPrinter()
    pp.pprint(param_grid)

    run_experiment(param_grid)