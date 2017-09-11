import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utilities.datacontainers import Summary, SummarySmall
from utilities.numerical import moving_average


def plot_results(files, num_exp=1, legend=None):
    for file in files:
        if num_exp > 1:
            fps = ["%s_%d.json" % (file, idx) for idx in range(num_exp)]
        else:
            fps = ["%s.json" % (file)]
        summaries = [SummarySmall.load(fp) for fp in fps]
        rewards = np.array([np.array(summary.average_episode_reward()) for summary in summaries])

        mean = np.mean(rewards, axis=0)
        ma_mean = moving_average(mean)
        std = np.std(rewards, axis=0)
        upper = moving_average(mean + std)
        lower = moving_average(mean - std)

        plt.fill_between(range(len(upper)), lower, upper, alpha=0.3)
        plt.plot(range(len(ma_mean)), ma_mean)

    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    if legend is None:
        legend = ["%s" % file.split('/')[-1].split('.')[-2] for file in files]
    plt.legend(legend)
    plt.show()