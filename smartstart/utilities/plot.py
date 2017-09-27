"""Module for plotting results

This module describes methods for generating a few predefined plots. The
methods make it easy to plot multiple different experiments and average the
results of experiments with the same parameters.

The results are :class:`~smartstart.utilities.datacontainers.Summary` objects
saved as JSON strings.
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from smartstart.utilities.numerical import moving_average
from smartstart.utilities.datacontainers import Summary


def mean_reward_std_episode(summaries, ma_window=1, color=None, linestyle=None):
    """Plot mean reward with standard deviation per episode

    Parameters
    ----------
    summaries : :obj:`list` of :obj:`~smartstart.utilities.datacontainers.Summary`
        summaries to average and plot
    ma_window : :obj:`int`
        moving average window size (Default value = 1)
    color :
        color (Default value = None)
    linestyle :
        linestyle (Default value = None)

    """
    rewards = np.array([np.array(summary.average_episode_reward()) for summary in summaries])

    mean = np.mean(rewards, axis=0)
    ma_mean = moving_average(mean, ma_window)
    std = np.std(rewards, axis=0)
    upper = moving_average(mean + std)
    lower = moving_average(mean - std)

    plt.fill_between(range(len(upper)), lower, upper, alpha=0.3, color=color)
    plt.plot(range(len(ma_mean)), ma_mean, color=color, linestyle=linestyle, linewidth=1.)


def mean_reward_episode(summaries, ma_window=1, color=None, linestyle=None):
    """Plot mean reward per episode

    Parameters
    ----------
    summaries : :obj:`list` of :obj:`~smartstart.utilities.datacontainers.Summary`
        summaries to average and plot
    ma_window : :obj:`int`
        moving average window size (Default value = 1)
    color :
        color (Default value = None)
    linestyle :
        linestyle (Default value = None)

    """
    rewards = np.array([np.array(summary.average_episode_reward()) for summary in summaries])

    mean = np.mean(rewards, axis=0)
    ma_mean = moving_average(mean, ma_window)

    plt.plot(range(len(ma_mean)), ma_mean, color=color, linestyle=linestyle, linewidth=1.)


def steps_episode(summaries, ma_window=1, color=None, linestyle=None):
    """Plot number of steps per episode


    Parameters
    ----------
    summaries : :obj:`list` of :obj:`~smartstart.utilities.datacontainers.Summary`
        summaries to average and plot
    ma_window : :obj:`int`
        moving average window size (Default value = 1)
    color :
        color (Default value = None)
    linestyle :
        linestyle (Default value = None)

    """
    steps = np.array([np.array(summary.steps_episode()) for summary in summaries])

    mean = np.mean(steps, axis=0)
    ma_mean = moving_average(mean, ma_window)

    plt.plot(range(len(ma_mean)), ma_mean, color=color, linestyle=linestyle, linewidth=1.)


labels = {
    mean_reward_std_episode: ["Episode", "Average Reward"],
    mean_reward_episode: ["Episode", "Average Reward"],
    steps_episode: ["Episode", "Steps per Episode"]
}


def plot_summary(files, plot_type, ma_window=1, title=None, legend=None,
                 output_dir=None, colors=None, linestyles=None,
                 format="eps", baseline=None):
    """Main plot function to be used

    The files parameter can be a list of files or a list of
    :obj:`~smartstart.utilities.datacontainers.Summary` objects that you
    want to compare in a single plot. A single file or single
    :obj:`~smartstart.utilities.datacontainers.Summary` can also be provided.
    Please read the instructions below when supplying a list of files.

    The files list provided must contain filenames without the ``.json``
    extension. For example: ``['file/path/to/experiment']`` is correct but ``[
    'file/path/to/experiment.json']`` not! The reason for this is when the
    folder contains multiple summary files from the same experiment (same
    parameters) it will use all the files and average them. For example when
    the folder contains the following three files ``[
    'file/path/to/experiment_1.json', 'file/path/to/experiment_2.json',
    'file/path/to/experiment_3.json']``. By providing ``[
    'file/path/to/experiment']`` all three summaries will be loaded and averaged.

    Note:
        The entries in files have to be defined without ``.json`` at the end.

    Note:
        Don't forget to run the show_plot() function after initializing the
        plots. Else nothing will be rendered on screen

    Parameters
    ----------
    files : :obj:`list` of :obj:`str` or :obj:`list` of
    :obj:`~smartstart.utilities.datacontainers.Summary`
        Option 1: each entry is the filepath to a saved summary without
        ``.json`` at the end. Option 2: each entry is a Summary object.
    plot_type :
        one of the plot functions defined in this module
    ma_window : :obj:`int`
        moving average filter window size (Default value = 10)
    title : :obj:`str`
        title of the plot, is also used as filename (Default value = None)
    legend : :obj:`list` of :obj:`str`
        one entry per entry in files (Default value = None)
    output_dir : :obj:`str`
        if not None the plot will be saved in this directory (Default value =
        None)
    colors : :obj:`list`
        one entry per entry in files (Default value = None)
    linestyles : :obj:`list`
        one entry per entry in files (Default value = None)
    format : :obj:`str`
        output format when saving plot (Default value = "eps")
    baseline : :obj:`float`
        plotting a dotted horizontal line as baseline (Default value = None)
    """
    if colors is not None:
        assert len(colors) == len(files)
    if linestyles is not None:
        assert len(linestyles) == len(files)
    if type(files) is not list:
        files = [files]

    plt.figure()
    xmax = 0
    for file in files:
        if type(file) is Summary:
            summaries = [file]
        else:
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
    """Helper method for saving plots

    Parameters
    ----------
    output_dir : :obj:`str`
        directory where the plot is saved
    title : :obj:`str`
        filename of saved plot
    format : :obj:`str`
        file format (Default value = "eps")

    Raises
    ------
    Exception
        Please give a title when saving a figure.
    """
    sns.set_context("paper")
    if title is None:
        raise Exception("Please give a title when saving a figure.")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = title.replace(" ", "_")
    fp = os.path.join(output_dir, filename + "." + format)
    plt.savefig(fp,
                format=format,
                dpi=1200,
                bbox_inches="tight")


def show_plot():
    """Render the plots on screen

    Must be run after initializing the plots to actually show them on screen.
    """
    plt.show()