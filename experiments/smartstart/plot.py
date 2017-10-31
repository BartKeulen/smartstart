import os

import seaborn as sns

from smartstart.algorithms.tdlearning import TDLearning
from smartstart.utilities.plot import plot_summary, mean_reward_episode, \
    mean_reward_std_episode, steps_episode, show_plot
from smartstart.utilities.utilities import get_data_directory

# sns.set_context("paper")

directory = get_data_directory(__file__)
mazes = ["GridWorldEasy"]
baselines = [31]
algos = ["QLearning", "SARSA", "SARSALambda"]
exploration_strategies = [TDLearning.E_GREEDY, TDLearning.BOLTZMANN, TDLearning.COUNT_BASED, TDLearning.UCT]
use_smart_start = [True, False]
colors = [color for color in sns.color_palette("husl", 4) for _ in (0, 1)]
linestyles = ['-', ':']*4

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

        plot_summary(files,
                     plot_type=mean_reward_episode,
                     train_bool=False,
                     title=title + "_mean_reward_episode",
                     legend=legend,
                     colors_in=colors,
                     linestyles_in=linestyles,
                     baseline=1/baseline)

show_plot()