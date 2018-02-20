import logging
import os

import numpy as np

from smartstart.agents.qlearning import QLearning
from smartstart.agents.smartstart import SmartStart
from smartstart.agents.valueiteration import ValueIteration
from smartstart.utilities.counter import Counter
from smartstart.environments.gridworld import GridWorld
import smartstart.rl as rl
from smartstart.utilities.experimenter import run_experiment

# Set logging level to warning
logging.basicConfig(level=logging.WARNING, format='%(name)s - %(levelname)s - %(message)s')

# Set directory to save the results
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# Set training parameters
rl.RENDER = False
rl.RENDER_EPISODE = False
rl.RENDER_TEST = False
rl.MAX_STEPS_EPISODE = 500
rl.MAX_STEPS = 50000
rl.TEST_FREQ = 500


# Define the task function for the experiment
def task(params):
    # Reset the seed for random number generation
    np.random.seed()

    # Create environment
    env = GridWorld.generate(GridWorld.MEDIUM)

    # Initialize Q-Learning agent, see class for available parameters
    state_action_shape = (env.h, env.w, env.num_actions)
    state_action_values = np.zeros(state_action_shape)
    counter = Counter(state_action_shape)
    agent = QLearning(state_action_values, counter,
                      exploration_strategy=QLearning.UCB1)

    # Initialize Smart Start agent
    if params['use_smart_start']:
        value_iteration = ValueIteration(state_action_shape)
        agent = SmartStart(agent, value_iteration, counter, c_ss=0.1, eta=0.8)

    # Train the agent, summary contains training data. Make sure to set the
    # rendering and printing to False when multiple experiments run in
    # parallel. Else it will consume a lot of computation power.
    summary = rl.train(env, agent)

    # Save the summary. The post_fix parameter can be used to create a unique
    #  file name.
    summary.save(directory=data_dir, post_fix="%.2d" % params['run'])


# Define a parameter grid that can be supplied to the run_experiment method
param_grid = {
    'task': task,
    'num_exp': 2,
    'use_smart_start': [True, False]
}

run_experiment(param_grid, n_processes=-1)  # n_processes=-1 uses all available cores
