import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utilities.numerical import moving_average
from utilities.serializer import deserialize


def plot_results(files, num_exp, legend=None):
    for file in files:
        fps = ["%s_%d.bin" % (file, idx) for idx in range(num_exp)]
        summaries = [deserialize(fp) for fp in fps]
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