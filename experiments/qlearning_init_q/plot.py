import os

import matplotlib.pyplot as plt

from experiments.qlearning_init_q.experiment import num_exp, init_q_values, \
    exploration_strategies
from smartstart.utilities.plot import mean_reward_episode

maze = "GridWorldEasy"
algo = "QLearning"

directory = "/home/bartkeulen/repositories/smartstart/data/qlearning_init_q"

for exploration_strategy in exploration_strategies:
    files = []
    legend = []
    for init_q in init_q_values:
        legend.append(r'init_q = %.7f' % init_q)
        files.append(os.path.join(directory, "%s_%s_exploration=%s_init_q=%.7f" % (algo, maze, exploration_strategy, init_q)))
    title = "%s %s %s init_q" % (algo, maze, exploration_strategy)
    mean_reward_episode(files, num_exp=num_exp, title=title, legend=legend)

plt.show()