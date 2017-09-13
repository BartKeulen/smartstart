import os

import matplotlib.pyplot as plt
import seaborn as sns

from utilities.plot import plot_mean

from experiments.test_experiment.experiment import alphas, directory, num_exp

maze = "GridWorldEasy"
algo = "QLearning"

files = []
legend = []
for alpha in alphas:
    legend.append(r'alpha = %.2f' % alpha)
    files.append(os.path.join(directory, "%s_%s_alpha=%.2f" % (algo, maze, alpha)))
title = "%s %s alpha" % (algo, maze)
plot_mean(files, num_exp=num_exp, title=title, legend=legend)

plt.show()