import os

import seaborn as sns

from smartstart.algorithms.tdlearning import TDLearning
from smartstart.utilities.plot import plot_summary, mean_reward_episode, \
    mean_reward_std_episode, steps_episode, show_plot
from smartstart.utilities.utilities import get_data_directory

sns.set_context("paper")

directory = "/home/bartkeulen/repositories/smartstart/data/tmp/smartstart" \
            "/GridWorldEasy"

# mazes = ["GridWorldEasy", "GridWorldMedium"]
# baselines = [31, 72]
# algos = ["QLearning", "SARSA", "SARSALambda"]
# exploration_strategies = ["E-Greedy", "Count-Based", "Boltzmann", "None"]
# use_smart_start = [True, False]

directory = get_data_directory(__file__)
mazes = ["GridWorldEasy"]
baselines = [31]
algos = ["QLearning"]
exploration_strategies = [TDLearning.E_GREEDY, TDLearning.UCT]
use_smart_start = [True, False]

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

        # plot_summary(files,
        #              plot_type=steps_episode,
        #              ma_window=10,
        #              title=title + "_steps_episode",
        #              legend=legend,
        #              baseline=baseline)
        #
        # plot_summary(files,
        #              plot_type=mean_reward_std_episode,
        #              ma_window=10,
        #              title=title + "_mean_reward_std_episode",
        #              baseline=1/baseline)

        plot_summary(files,
                     plot_type=mean_reward_episode,
                     train_bool=False,
                     title=title + "_mean_reward_episode",
                     legend=legend,
                     baseline=1/baseline)

show_plot()