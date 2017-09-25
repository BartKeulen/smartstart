import time

import numpy as np
from utilities.experimenter import run_experiment

from smartexploration.algorithms import QLearning
from smartexploration.algorithms import SARSA, SARSALambda
from smartexploration.algorithms import TDLearning
from smartexploration.environments.gridworld import GridWorld
from smartexploration.smartexploration.smartexploration import SmartStart
from smartexploration.utilities.utilities import get_data_directory

directory = get_data_directory(__file__)

gridworlds = [GridWorld.EASY, GridWorld.MEDIUM, GridWorld.HARD]
algorithms = [QLearning, SARSA, SARSALambda]
exploration_strategies = [TDLearning.COUNT_BASED, TDLearning.E_GREEDY, TDLearning.BOLTZMANN, TDLearning.NONE]
use_smart_start = [True, False]
num_exp = 25

cur_time = time.time()


def task(params):
    np.random.seed()

    if params['gridworld'] == GridWorld.EASY:
        num_episodes = 1000
        max_steps = 1000
        exploration_steps = 100
    elif params['gridworld'] == GridWorld.MEDIUM:
        num_episodes = 2500
        max_steps = 2500
        exploration_steps = 250
    elif params['gridworld'] == GridWorld.HARD:
        num_episodes = 10000
        max_steps = 10000
        exploration_steps = 1000
    else:
        raise Exception("Invalid gridworld type")

    env = GridWorld.generate(params['gridworld'])

    if params['use_smart_start']:
        agent = SmartStart(params['algorithm'],
                           env,
                           alpha=0.1,
                           gamma=0.99,
                           epsilon=0.05,
                           num_episodes=num_episodes,
                           max_steps=max_steps,
                           exploration=params['exploration_strategy'],
                           exploration_steps=exploration_steps,
                           exploitation_param=0.)
    else:
        agent = params['algorithm'](env,
                                    alpha=0.1,
                                    gamma=0.99,
                                    epsilon=0.05,
                                    num_episodes=num_episodes,
                                    max_steps=max_steps,
                                    exploration=params['exploration_strategy'])

    post_fix = "exploration=%s_%d" % (params['exploration_strategy'], params['run'])

    summary = agent.train(render=False, render_episode=False, print_results=False)

    # summary.save(directory, post_fix)
    summary.save_to_gcloud("smartexploration/%.0f/%s" % (cur_time, env.name), post_fix)


param_grid = {'task': task,
              'gridworld': gridworlds,
              'algorithm': algorithms,
              'exploration_strategy': exploration_strategies,
              'use_smart_start': use_smart_start,
              'num_exp': num_exp}


if __name__ == "__main__":
    run_experiment(param_grid)