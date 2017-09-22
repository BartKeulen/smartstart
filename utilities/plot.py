import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utilities.datacontainers import Summary
from utilities.numerical import moving_average


def mean_reward_std_episode(summaries, ma_window=10, color=None, linestyle=None):
    rewards = np.array([np.array(summary.average_episode_reward()) for summary in summaries])

    mean = np.mean(rewards, axis=0)
    ma_mean = moving_average(mean, ma_window)
    std = np.std(rewards, axis=0)
    upper = moving_average(mean + std)
    lower = moving_average(mean - std)

    plt.fill_between(range(len(upper)), lower, upper, alpha=0.3, color=color)
    plt.plot(range(len(ma_mean)), ma_mean, color=color, linestyle=linestyle, linewidth=1.)


def mean_reward_episode(summaries, ma_window=10, color=None, linestyle=None):
    rewards = np.array([np.array(summary.average_episode_reward()) for summary in summaries])

    mean = np.mean(rewards, axis=0)
    ma_mean = moving_average(mean, ma_window)

    plt.plot(range(len(ma_mean)), ma_mean, color=color, linestyle=linestyle, linewidth=1.)


def steps_episode(summaries, ma_window=10, color=None, linestyle=None):
    steps = np.array([np.array(summary.steps_episode()) for summary in summaries])

    mean = np.mean(steps, axis=0)
    ma_mean = moving_average(mean, ma_window)

    plt.plot(range(len(ma_mean)), ma_mean, color=color, linestyle=linestyle, linewidth=1.)


# def cumsum_reward_episode(summaries, ma_window=10, color=None, linestyle=None):
#     rewards = np.array([np.array(summary.average_episode_reward()) for summary in summaries])
#     cumsum_rewards = np.cumsum(rewards)
#
#     mean = np.mean(rewards, axis=0)
#     ma_mean = moving_average(mean, ma_window)
#
#     plt.plot(range(len(ma_mean)), ma_mean, color=color, linestyle=linestyle)


labels = {
    mean_reward_std_episode: ["Episode", "Average Reward"],
    mean_reward_episode: ["Episode", "Average Reward"],
    steps_episode: ["Episode", "Steps per Episode"]
}


def plot(files, plot_type, ma_window=10, title=None, legend=None, output_dir=None, colors=None, linestyles=None,
         format="eps", baseline=None):
    if colors is not None:
        assert len(colors) == len(files)
    if linestyles is not None:
        assert len(linestyles) == len(files)

    plt.figure()
    xmax = 0
    for file in files:
        fps = glob.glob("%s*.json" % file)

        summaries = [Summary.load(fp) for fp in fps]

        xmax = max(xmax, len(summaries[0]))

        color, linestyle = None, None
        if colors is not None:
            color = colors.pop()
        if linestyles is not None:
            linestyle = linestyles.pop()

        plot_type(summaries, ma_window, color, linestyle)

    if baseline is not None:
        plt.hlines(y=baseline, xmin=0, xmax=xmax, color="black", linestyle="dotted")

    if title is not None and output_dir is None:
        plt.title(title)
    plt.autoscale(enable=True, axis='x', tight=True)
    x_label, y_label = labels[plot_type]
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if legend is not None:
        plt.legend(legend)

    if output_dir:
        save_plot(output_dir, title, format)


def save_plot(output_dir, title, format="eps"):
    sns.set_context("paper")
    if title is None:
        raise Exception("Please give a title when saving a figure.")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fp = os.path.join(output_dir, title + "." + format)
    plt.savefig(fp,
                format=format,
                dpi=1200,
                bbox_inches="tight")