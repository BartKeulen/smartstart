import os

import seaborn as sns

from smartstart.algorithms import QLearning
from smartstart.utilities.plot import plot_summary, mean_reward_episode, \
    mean_reward_std_episode, steps_episode, show_plot
from smartstart.utilities.utilities import get_data_directory

sns.set_context("paper")

# directory = get_data_directory(__file__)
directory = '/home/bart/Projects/smartstart/data/smartstart/'
# mazes = ["GridWorldEasy", "GridWorldMedium", "GridWorldHard"]
# baselines = [31, 72, 138]
mazes = ["GridWorldHard"]
baselines = [138]
algos = ["QLearning", "SARSA"]
exploration_strategies = [QLearning.E_GREEDY, QLearning.BOLTZMANN, QLearning.COUNT_BASED, QLearning.UCB1]
use_smart_start = [True, False]
colors = [color for color in sns.color_palette("husl", 4) for _ in (0, 1)]
linestyles = ['-', '-.']*4

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
                files.append(os.path.join(directory, maze, filename))
                legendname += "%s" % exploration_strategy
                legend.append(legendname)

        import pdb
        pdb.set_trace()

        title = "%s_%s" % (algo, maze)

        plot_summary(files,
                     plot_type=mean_reward_episode,
                     train_bool=False,
                     title=title + "_mean_reward_episode",
                     legend=legend,
                     colors=colors.copy(),
                     linestyles=linestyles.copy(),
                     baseline=1/baseline,
                     # output_dir='/home/bart/IHMC/thesis/img/',
                     format='pdf')

show_plot()