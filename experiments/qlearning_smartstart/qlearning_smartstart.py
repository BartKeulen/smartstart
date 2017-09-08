import os

import numpy as np

from algorithms.qlearning import QLearning
from algorithms.smartstart import SmartStart
from environments.gridworld import GridWorldVisualizer, GridWorld
from utilities.experimenter import run_experiment
from utilities.plot import plot_results
from utilities.utilities import get_data_directory

directory = get_data_directory(__file__)

maze_type = GridWorld.EASY
maze = 'GridWorldEasy'
num_exp = 5
num_episodes = 1000
max_steps = 500


def task(params):
    np.random.seed()

    env = GridWorld.generate(maze_type)
    env.visualizer.add_visualizer(GridWorldVisualizer.VALUE_FUNCTION,
                                  GridWorldVisualizer.DENSITY,
                                  GridWorldVisualizer.CONSOLE)

    if params['use_smart_start']:
        smart_start = SmartStart(env, alpha=0.)
        post_fix = "smartstart_%d" % params['run']
    else:
        smart_start = None
        post_fix = "%d" % params['run']

    agent = QLearning(env, alpha=0.3, num_episodes=num_episodes, max_steps=max_steps, smart_start=smart_start)
    summary = agent.train(render=False, render_episode=False, print_results=False)

    summary.save(directory=directory, post_fix=post_fix)


param_grid = {'task': task, 'use_smart_start': [True, False], 'num_exp': num_exp}

run_experiment(param_grid)

files = [os.path.join(directory, 'QLearning_%s' % maze),
         os.path.join(directory, 'QLearning_%s_smartstart' % maze)]
plot_results(files, num_exp, [r'Q-Learning', r'Q-Learning smartstart'])

