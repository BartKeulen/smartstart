import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utilities.datacontainers import Summary
from utilities.numerical import moving_average


def plot_mean_std(files, num_exp=1, ma_window=10, title=None, legend=None):
    plt.figure()
    for file in files:
        if num_exp > 1:
            fps = ["%s_%d.json" % (file, idx) for idx in range(num_exp)]
        else:
            fps = ["%s_0.json" % (file)]
        summaries = [Summary.load(fp) for fp in fps]
        rewards = np.array([np.array(summary.average_episode_reward()) for summary in summaries])

        mean = np.mean(rewards, axis=0)
        ma_mean = moving_average(mean, ma_window)
        std = np.std(rewards, axis=0)
        upper = moving_average(mean + std)
        lower = moving_average(mean - std)

        plt.fill_between(range(len(upper)), lower, upper, alpha=0.3)
        plt.plot(range(len(ma_mean)), ma_mean)

    if title is not None:
        plt.title(title)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    if legend is None:
        legend = ["%s" % file.split('/')[-1].split('.')[-2] for file in files]
    plt.legend(legend)


def plot_mean(files, num_exp=1, ma_window=10, title=None, legend=None):
    plt.figure()
    for file in files:
        if num_exp > 1:
            fps = ["%s_%d.json" % (file, idx) for idx in range(num_exp)]
        else:
            fps = ["%s_0.json" % (file)]
        summaries = [Summary.load(fp) for fp in fps]
        rewards = np.array([np.array(summary.average_episode_reward()) for summary in summaries])

        mean = np.mean(rewards, axis=0)
        ma_mean = moving_average(mean, ma_window)

        plt.plot(range(len(ma_mean)), ma_mean)

    if title is not None:
        plt.title(title)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    if legend is None:
        legend = ["%s" % file.split('/')[-1].split('.')[-2] for file in files]
    plt.legend(legend)