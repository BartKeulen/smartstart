import os

import numpy as np
from utilities.utilities import get_data_directory

from smartstart.algorithms import QLearning
from smartstart.algorithms import TDLearning
from smartstart.environments.gridworld import GridWorld
from smartstart.smartexploration.smartexploration import generate_smartstart_object
from smartstart.utilities.plot import mean_reward_std_episode

directory = get_data_directory(__file__)

maze_type = [GridWorld.EASY]
algorithms = [QLearning]
exploration_strategies = [TDLearning.NONE]
use_smart_start = [True]
# maze_type = [GridWorld.EASY, GridWorld.MEDIUM]
# algorithms = [QLearning, SARSA, SARSALamba]
# exploration_strategies = [TDLearning.E_GREEDY, TDLearning.BOLTZMANN, TDLearning.NONE]
# use_smart_start = [True, False]
num_exp = 25


def task(params):
    np.random.seed()

    if params['maze_type'] == GridWorld.EASY:
        num_episodes = 1000
        max_steps = 1000
        exploration_steps = 100
    elif params['maze_type'] == GridWorld.MEDIUM:
        num_episodes = 2500
        max_steps = 2500
        exploration_steps = 250
    elif params['maze_type'] == GridWorld.HARD:
        num_episodes = 10000
        max_steps = 10000
        exploration_steps = 500
    else:
        raise Exception("Not a valid maze type for this experiment")

    env = GridWorld.generate(params['maze_type'])

    if params['use_smart_start']:
        agent = generate_smartstart_object(params['algorithm'],
                                           env,
                                           alpha=0.3,
                                           num_episodes=num_episodes,
                                           max_steps=max_steps,
                                           exploration=params['exploration_strategy'],
                                           exploration_steps=exploration_steps)
    else:
        agent = params['algorithm'](env,
                                    alpha=0.3,
                                    num_episodes=num_episodes,
                                    max_steps=max_steps,
                                    exploration=params['exploration_strategy'])

    if params['exploration_strategy'] == TDLearning.E_GREEDY:
        post_fix = "egreedy_"
    elif params['exploration_strategy'] == TDLearning.BOLTZMANN:
        post_fix = "boltzmann_"
    else:
        post_fix = ""

    post_fix += "%d" % params['run']

    summary = agent.train(render=False, render_episode=False, print_results=False)

    summary.save(directory=directory, post_fix=post_fix)


param_grid = {'task': task,
              'maze_type': maze_type,
              'algorithm': algorithms,
              'exploration_strategy': exploration_strategies,
              'use_smart_start': use_smart_start,
              'num_exp': num_exp}

# run_experiment(param_grid)

maze = "GridWorldEasy"
files = [os.path.join(directory, 'QLearning_%s' % maze),
         os.path.join(directory, 'QLearning_%s_egreedy' % maze),
         # os.path.join(directory, 'QLearning_%s_boltzmann' % maze),
         os.path.join(directory, 'SmartStart_QLearning_%s_egreedy' % maze)]
         # os.path.join(directory, 'SmartStart_QLearning_%s_boltzmann' % maze)]
mean_reward_std_episode(files, 5, [r'None', r'$\epsilon$-greedy', r'SmartStart $\epsilon$-greedy'])

