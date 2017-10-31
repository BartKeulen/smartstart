import random
import time

import numpy as np

from smartstart.utilities.experimenter import run_experiment
from smartstart.algorithms import QLearning
from smartstart.algorithms import SARSA, SARSALambda
from smartstart.algorithms import TDLearning
from smartstart.environments.gridworld import GridWorld
from smartstart.smartexploration.smartexploration import generate_smartstart_object
from smartstart.utilities.utilities import get_data_directory

directory = get_data_directory(__file__)

cur_time = time.time()


def task(params):
    np.random.seed()
    random.seed()

    env = GridWorld.generate(GridWorld.EASY)

    kwargs = {
        'alpha': 0.1,
        'gamma': 0.99,
        'epsilon': 0.05,
        'num_episodes': 250,
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

    summary = agent.train(test_freq=10, render=False, render_episode=False, print_results=False)

    summary.save(directory, post_fix)

    # summary.save_to_gcloud(bucket_name='drl-data',
    #                        directory="smartstart/%.0f/%s" % (cur_time,
    #                                                          env.name),
    #                        post_fix=post_fix)


algorithms = [QLearning]
exploration_strategies = [TDLearning.E_GREEDY, TDLearning.UCT]
use_smart_start = [True, False]
num_exp = 5

param_grid = {'task': task,
              'algorithm': algorithms,
              'exploration_strategy': exploration_strategies,
              'use_smart_start': use_smart_start,
              'num_exp': num_exp}


if __name__ == "__main__":
    run_experiment(param_grid, n_processes=4)