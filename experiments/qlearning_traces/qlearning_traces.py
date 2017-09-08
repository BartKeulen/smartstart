import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from environments.gridworld import GridWorldVisualizer, GridWorld
from algorithms.qlearning import QLearning
from algorithms.qlearninglambda import QLearningLambda
from smartstart.smartstart_file import SmartStart
from utilities.utilities import get_data_directory
from utilities.experimenter import run_experiment
from utilities.plot import plot_results
from utilities.scheduler import LinearScheduler


directory = get_data_directory(__file__)


maze_type = GridWorld.HARD
num_exp = 5
num_episodes = 2500
max_steps = 10000


def task(params):
    np.random.seed()

    env = GridWorld.generate(maze_type)

    smart_start = SmartStart(env, exploration_steps=1000, alpha=0.)
    ss_scheduler = LinearScheduler(1000, 1500)

    if params['use_traces'] and params['smart_start']:
        agent = QLearningLambda(env,
                                alpha=0.3,
                                num_episodes=num_episodes,
                                max_steps=max_steps,
                                smart_start=smart_start,
                                ss_scheduler=ss_scheduler)
        post_fix = "smartstart_%d" % params['run']
    elif params['use_traces']:
        agent = QLearningLambda(env,
                                alpha=0.3,
                                num_episodes=num_episodes,
                                max_steps=max_steps)
        post_fix = "%d" % params['run']
    elif params['smart_start']:
        agent = QLearning(env,
                          alpha=0.3,
                          num_episodes=num_episodes,
                          max_steps=max_steps,
                          smart_start=smart_start,
                          ss_scheduler=ss_scheduler)
        post_fix = "smartstart_%d" % params['run']
    else:
        agent = QLearning(env,
                          alpha=0.3,
                          num_episodes=num_episodes,
                          max_steps=max_steps)
        post_fix = "%d" % params['run']

    summary = agent.train(print_results=False)

    summary.save(directory=directory, post_fix=post_fix)


param_grid = {'task': task, 'use_traces': [True, False], 'smart_start': [True, False], 'num_exp': num_exp}

run_experiment(param_grid)

# Plot results or exit
maze = 'GridWorldHard'
files = [os.path.join(directory, 'QLearning_%s' % maze),
         os.path.join(directory, 'QLearningLambda_%s' % maze),
         os.path.join(directory, 'QLearning_%s_smartstart' % maze),
         os.path.join(directory, 'QLearningLambda_%s_smartstart' % maze)]
plot_results(files, num_exp, [r'Q-Learning', r'Q-Learning($\lambda$)', r'Q-Learning SmartStart', r'Q-Learning($\lambda$) SmartStart'])
