import os

import seaborn as sns

from smartstart.utilities.plot import plot_summary, mean_reward_episode, \
    mean_reward_std_episode, steps_episode

sns.set_context("paper")

directory = "/home/bartkeulen/repositories/smartstart/data/smartstart"

mazes = ["GridWorldEasy", "GridWorldMedium"]
baselines = [31, 72]
algos = ["QLearning", "SARSA", "SARSALamba"]
exploration_strategies = ["E-Greedy", "Count-Based", "Boltzmann", "None"]
use_smart_start = [True, False]

# mazes = ["GridWorldMedium"]
# algos = ["QLearning"]
exploration_strategies = ["E-Greedy"]

# colors = np.kron(np.asarray(sns.color_palette("husl", len(exploration_strategies))), np.ones((2, 1)))
# colors = [tuple(row) for row in colors]
# linestyles = ["solid", "dotted"] * len(exploration_strategies)
colors = ["blue", "green"]

for maze, baseline in zip(mazes, baselines):
    for algo in algos:
        files = []
        legend = []
        for exploration_strategy in exploration_strategies:
            for smart_start in use_smart_start:
                filename = ""
                legendname = ""
                if smart_start:
                    filename += "SmartStart_"
                    legendname += "SmartStart "
                filename += "%s_%s_exploration=%s" % (algo, maze, exploration_strategy)
                files.append(os.path.join(directory, filename))
                legendname += "%s" % exploration_strategy
                legend.append(legendname)
        title = "%s_%s" % (algo, maze)
        plot_summary(files, plot_type=steps_episode, ma_window=1, title=title + "_steps_episode", legend=legend,
                     output_dir="/home/bartkeulen/IHMC/thesis/img/tmp", baseline=baseline)

        plot_summary(files, plot_type=mean_reward_std_episode, ma_window=1, title=title + "_mean_reward_std_episode",
                     legend=legend, output_dir="/home/bartkeulen/IHMC/thesis/img/tmp", format="png", baseline=1/baseline)

        plot_summary(files, plot_type=mean_reward_episode, ma_window=1, title=title + "_mean_reward_episode", legend=legend,
                     output_dir="/home/bartkeulen/IHMC/thesis/img/tmp", baseline=1/baseline)

# plt.show()