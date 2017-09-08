import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from environments.gridworld import GridWorld, GridWorldVisualizer
from algorithms.qlearning import QLearning
from utilities.utilities import get_data_directory
from utilities.serializer import deserialize
from utilities.numerical import moving_average

directory = get_data_directory(__file__)

init_q_values = [0., 0.001, 0.01, 0.1, 1.]
num_exp = 5
num_episodes = 1000
max_steps = 25000
run_experiment = True
plot_results = True
colors = ['b', 'g', 'r', 'c', 'y']

# Execute experiment
if run_experiment:
    for init_q in init_q_values:
        for i in range(num_exp):
            print("\nStart experiment. Init q-value: %.3f, exp num: %d" % (init_q, i))

            np.random.seed()

            env = GridWorld.generate(GridWorld.HARD)

            agent = QLearning(env, num_episodes=num_episodes, max_steps=max_steps, init_q_value=init_q)

            summary = agent.train(render=False)

            post_fix = "%.3f_%d" % (init_q, i)
            summary.save(directory=directory, post_fix=post_fix)

# Plot results or exit
if plot_results:
    for i, init_q in enumerate(init_q_values):
        fps = [os.path.join(directory, "QLearning_GridWorldHard_%.3f_%d.bin" % (init_q, idx)) for idx in range(num_exp)]
        summaries = [deserialize(fp) for fp in fps]
        rewards = np.array([np.array(summary.average_episode_reward()) for summary in summaries])

        mean = np.mean(rewards, axis=0)
        ma_mean = moving_average(mean)
        std = np.std(rewards, axis=0)
        upper = moving_average(mean + std)
        lower = moving_average(mean - std)

        plt.fill_between(range(len(upper)), lower, upper, facecolor=colors[i], alpha=0.3)
        plt.plot(range(len(ma_mean)), ma_mean, colors[i])

    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    legend = ["init q = %.3f" % init_q for init_q in init_q_values]
    plt.legend(legend)
    plt.show()