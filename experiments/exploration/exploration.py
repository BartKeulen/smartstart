import os

import numpy as np
import matplotlib.pyplot as plt

from algorithms.tdlearning import TDLearning
from algorithms.qlearning import QLearning
from algorithms.sarsa import SARSA, SARSALamba
from algorithms.smartstart import SmartStart
from environments.gridworld import GridWorldVisualizer, GridWorld
from utilities.experimenter import run_experiment
from utilities.plot import plot_mean_std
from utilities.utilities import get_data_directory

directory = get_data_directory(__file__)

maze_type = [GridWorld.EASY, GridWorld.MEDIUM]
algorithms = [QLearning, SARSA, SARSALamba]
exploration_strategies = [TDLearning.COUNT_BASED, TDLearning.E_GREEDY, TDLearning.BOLTZMANN, TDLearning.NONE]
num_exp = 10


def task(params):
    np.random.seed()

    if params['maze_type'] == GridWorld.EASY:
        num_episodes = 1000
        max_steps = 1000
    elif params['maze_type'] == GridWorld.MEDIUM:
        num_episodes = 2500
        max_steps = 2500
    elif params['maze_type'] == GridWorld.HARD:
        num_episodes = 10000
        max_steps = 10000
    else:
        raise Exception("Not a valid maze type for this experiment")

    env = GridWorld.generate(params['maze_type'])

    agent = params['algorithm'](env,
                                alpha=0.3,
                                num_episodes=num_episodes,
                                max_steps=max_steps,
                                exploration=params['exploration_strategy'])

    if params['exploration_strategy'] == TDLearning.COUNT_BASED:
        post_fix = "countbased_"
    elif params['exploration_strategy'] == TDLearning.E_GREEDY:
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
              'num_exp': num_exp}

# run_experiment(param_grid)

mazes = ["GridWorldEasy", "GridWorldMedium"]
algos = ["QLearning", "SARSA", "SARSALambda"]
for maze in mazes:
    for algo in algos:
        files = [os.path.join(directory, "%s_%s" % (algo, maze)),
                 os.path.join(directory, "%s_%s_egreedy" % (algo, maze)),
                 os.path.join(directory, "%s_%s_boltzmann" % (algo, maze)),
                 os.path.join(directory, "%s_%s_countbased" % (algo, maze))]
        legend = [r'None', r'$\epsilon$-greedy', r'Boltzmann', r'Count-Based']
        plot_mean_std(files, num_exp, legend)

plt.show()


