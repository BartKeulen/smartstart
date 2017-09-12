import os

import numpy as np
import matplotlib.pyplot as plt

from algorithms.tdlearning import TDLearning
from algorithms.qlearning import QLearning
from algorithms.sarsa import SARSA, SARSALamba
from algorithms.smartstart import SmartStart
from environments.gridworld import GridWorldVisualizer, GridWorld
from utilities.experimenter import run_experiment
from utilities.plot import plot_mean_std, plot_mean
from utilities.utilities import get_data_directory

directory = get_data_directory(__file__)

maze_type = [GridWorld.EASY, GridWorld.MEDIUM]
alphas = [0.01, 0.05, 0.1, 0.25, 0.5, 1.]
epsilons = [0.01, 0.05, 0.1, 0.25]
gammas = [0.5, 0.9, 0.95, 0.99, 0.999]
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

    agent = QLearning(env,
                      alpha=params['alpha'],
                      gamma=params['gamma'],
                      num_episodes=num_episodes,
                      max_steps=max_steps,
                      exploration=TDLearning.E_GREEDY,
                      epsilon=params['epsilon'])

    post_fix = "alpha=%.2f_gamma=%.3f_epsilon=%.2f_%d" % (params['alpha'], params['gamma'], params['epsilon'], params['run'])

    summary = agent.train(render=False, render_episode=False, print_results=False)

    summary.save(directory=directory, post_fix=post_fix)


param_grid = {'task': task,
              'maze_type': maze_type,
              'alpha': alphas,
              'epsilon': epsilons,
              'gamma': gammas,
              'num_exp': num_exp}

# run_experiment(param_grid)

mazes = ["GridWorldEasy", "GridWorldMedium"]
algo = "QLearning"

for maze in mazes:
    files = []
    legend = []
    for alpha in alphas:
        legend.append(r'alpha = %.2f' % alpha)
        files.append(os.path.join(directory, "%s_%s_alpha=%.2f_gamma=0.990_epsilon=0.05" % (algo, maze, alpha)))
    title = "%s %s alpha" % (algo, maze)
    plot_mean(files, num_exp, title, legend)

    files = []
    legend = []
    for gamma in gammas:
        legend.append(r'gamma = %.3f' % gamma)
        files.append(os.path.join(directory, "%s_%s_alpha=0.10_gamma=%.3f_epsilon=0.05" % (algo, maze, gamma)))
    title = "%s %s gamma" % (algo, maze)
    plot_mean(files, num_exp, title, legend)

    files = []
    legend = []
    for epsilon in epsilons:
        legend.append(r'$\epsilon$ = %.2f' % epsilon)
        files.append(os.path.join(directory, "%s_%s_alpha=0.10_gamma=0.990_epsilon=%.2f" % (algo, maze, epsilon)))
    title = r"%s %s $\epsilon$" % (algo, maze)
    plot_mean(files, num_exp, title, legend)


plt.show()