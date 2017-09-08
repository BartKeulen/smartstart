import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from environments.gridworld import GridWorldVisualizer, GridWorld
from algorithms.qlearning import QLearning
from smartstart.smartstart_file import SmartStart
from utilities.utilities import get_data_directory
from utilities.serializer import deserialize
from utilities.numerical import moving_average


directory = get_data_directory(__file__)

maze_type = GridWorld.EASY
smart_start_bool = [True, False]
num_exp = 5
num_episodes = 1000
max_steps = 25000
run_experiment = True
plot_results = True
colors = ['b', 'g']



# Execute experiment
if run_experiment:
    for use_smart_start in smart_start_bool:
        for i in range(num_exp):
            np.random.seed()

            env = GridWorld.generate(maze_type)
            env.visualizer.add_visualizer(GridWorldVisualizer.VALUE_FUNCTION,
                                          GridWorldVisualizer.DENSITY,
                                          GridWorldVisualizer.CONSOLE)

            if use_smart_start:
                smart_start = SmartStart(env, alpha=0.)
            else:
                smart_start = None

            agent = QLearning(env, alpha=0.3, num_episodes=num_episodes, max_steps=max_steps, smart_start=smart_start)
            summary = agent.train(render=False, render_episode=False)

            post_fix = "%s_%d" % (use_smart_start, i)
            summary.save(directory=directory, post_fix=post_fix)

# Plot results or exit
if plot_results:
    for i, use_smart_start in enumerate(smart_start_bool):
        fps = [os.path.join(directory, "QLearning_GridWorldEasy_%s_%d.bin" % (use_smart_start, idx)) for idx in range(num_exp)]
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
    legend = ["smart_start = %s" % use_smart_start for use_smart_start in smart_start_bool]
    plt.legend(legend)
    plt.show()