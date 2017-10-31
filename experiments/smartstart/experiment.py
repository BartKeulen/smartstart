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
        'num_episodes': 500,
        'max_steps': 1000,
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

    post_fix = "exploration=%s_%d" % (params['exploration_strategy'], params['run'])

    summary = agent.train(test_freq=5, render=False, render_episode=False, print_results=False)

    summary.save_to_gcloud(bucket_name='smartstart',
                           directory="smartstart/%s" % env.name,
                           post_fix=post_fix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('environment', type=int, help='An integer for the GridWorld environment (0 = Easy, 1 = Medium, 2 = Hard, 3 = Extreme)')
    args = parser.parse_args()

    env = [args.environment]
    algorithms = [QLearning, SARSA, SARSALambda]
    exploration_strategies = [TDLearning.E_GREEDY, TDLearning.BOLTZMANN, TDLearning.COUNT_BASED, TDLearning.UCT]
    use_smart_start = [True, False]
    num_exp = 25

    param_grid = {'task': task,
                  'env': env,
                  'algorithm': algorithms,
                  'exploration_strategy': exploration_strategies,
                  'use_smart_start': use_smart_start,
                  'num_exp': num_exp}

    pp = pprint.PrettyPrinter()
    pp.pprint(param_grid)

    run_experiment(param_grid)