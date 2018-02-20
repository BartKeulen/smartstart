import logging

import numpy as np
import matplotlib.pyplot as plt

from smartstart.agents.qlearning import QLearning
from smartstart.agents.smartstart import SmartStart
from smartstart.agents.valueiteration import ValueIteration
from smartstart.utilities.counter import Counter
from smartstart.environments.gridworld import GridWorld
from smartstart.environments.gridworldvisualizer import GridWorldVisualizer
import smartstart.rl as rl

# Set logging level to info to see text
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

# Reset the seed for random number generation
np.random.seed()

# Set training parameters
rl.RENDER = False
rl.RENDER_EPISODE = True
rl.MAX_STEPS_EPISODE = 150
rl.MAX_STEPS = 25000
rl.TEST_FREQ = 500
rl.RENDER_TEST = False

# Create environment and visualizer
env = GridWorld.generate(GridWorld.EASY)
visualizer = GridWorldVisualizer(env)
visualizer.add_visualizer(GridWorldVisualizer.ALL)

# Initialize Q-Learning agent, see class for available parameters
state_action_shape = (env.h, env.w, env.num_actions)
state_action_values = np.zeros(state_action_shape)
counter = Counter(state_action_shape)
agent = QLearning(state_action_values, counter)

# Initialize SmartStart agent
value_iteration = ValueIteration(state_action_shape)
agent = SmartStart(agent, value_iteration, counter, c_ss=0.1, eta=0.8)

# Train the agent, summary contains training data
summary = rl.train(env, agent)

# Plot results
plt.figure()

train_iter = summary.get_train_iterations_in_training_steps()
train_average_reward = summary.get_train_average_reward()
plt.plot(train_iter, train_average_reward, label='Train')

test_iter = summary.get_test_iterations_in_training_steps()
test_average_reward = summary.get_test_average_reward()
plt.plot(test_iter, test_average_reward, label='Test')

plt.xlabel('Training steps')
plt.ylabel('Average reward')
plt.legend()
plt.show()