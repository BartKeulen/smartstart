import os

import numpy as np

from algorithms.tdlearning import TDLearning
from algorithms.qlearning import QLearning
from algorithms.sarsa import SARSA, SARSALamba
from algorithms.smartstart import SmartStart
from environments.gridworld import GridWorldVisualizer, GridWorld
from utilities.experimenter import run_experiment
from utilities.plot import plot_mean_std
from utilities.utilities import get_data_directory
from utilities.datacontainers import SummarySmall, Summary

directory = get_data_directory(__file__)

maze_type = [GridWorld.EASY, GridWorld.MEDIUM, GridWorld.HARD]
algorithms = [QLearning]
exploration_strategies = [TDLearning.E_GREEDY]
use_smart_start = [True]
num_exp = 2

raise Exception("Not implemented anymore, have to return two summaries after training for comparison.")


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
        agent = SmartStart(params['algorithm'],
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

    summary, summary_full = agent.train(render=False, render_episode=False, print_results=False)

    summary.save(directory=directory, post_fix="small_%d" % params['run'])
    summary_full.save(directory=directory, post_fix="full_%d" % params['run'])


param_grid = {'task': task,
              'maze_type': maze_type,
              'algorithm': algorithms,
              'exploration_strategy': exploration_strategies,
              'use_smart_start': use_smart_start,
              'num_exp': num_exp}

run_experiment(param_grid, n_processes=6)

