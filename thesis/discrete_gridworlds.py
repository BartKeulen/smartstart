import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from smartstart.algorithms import QLearning
from smartstart.utilities.datacontainers import Summary
from smartstart.utilities.plot import *

sns.set()


def get_summaries(fp_data, env, algo, exploration_strategies):
    summaries = []
    for exploration_strategy in exploration_strategies:
        for smart_start in [True, False]:
            summary = {'env': env,
                       'algo': algo,
                       'exploration_strategy': exploration_strategy,
                       'smart_start': smart_start}
            filename = ""
            if smart_start:
                filename += "SmartStart_"
            filename += "%s_%s_%s" % (algo, env, exploration_strategy)
            fp = os.path.join(fp_data, env, filename)
            fps = glob.glob("%s*.json" % fp)
            summary['summaries'] = [Summary.load(fp) for fp in fps]
            summaries.append(summary)
    return summaries


def plot_summaries(summaries, n_types, baseline, xlim):
    legend = []
    for summary in summaries:
        label = ""
        if summary['smart_start']:
            label += "Smart Start "
        label += summary['exploration_strategy']
        legend.append(label)

    colors = [color for color in sns.color_palette("husl", n_types) for _ in (0, 1)]
    linestyles = ['-', '-.'] * n_types

    plot_summary([summary['summaries'] for summary in summaries],
                 plot_type=mean_reward_training_steps,
                 train_bool=False,
                 legend=legend,
                 baseline=baseline,
                 xlim=xlim,
                 colors=colors.copy(),
                 linestyles=linestyles.copy())


def get_rise_time(summaries, baseline, epsilon):
    for summary_set in summaries:
        rewards = []
        steps = []
        for summary in summary_set['summaries']:
            rewards.append(summary.mean_reward_episode(False))
            steps.append(summary.iterations_as_training_steps(False))
        rewards = np.asarray(rewards)
        steps = np.asarray(steps)

        x = np.unique(np.concatenate(steps))
        y = np.asarray([np.interp(x, episode, reward) for episode, reward in zip(steps, rewards)])
        mean_y = np.mean(y, axis=0)
        std = np.std(y, axis=0)

        summary_set['max'] = np.max(mean_y)

        threshold_rise = baseline * epsilon
        idx = np.argmax(mean_y >= threshold_rise)

        if idx == 0:
            summary_set.update({'rise time': np.nan,
                                'mean': np.nan,
                                'std': np.nan})
        else:
            summary_set.update({'rise time': x[idx],
                                'mean': mean_y[idx],
                                'std': std[idx]})


def save_results(fp, results, fieldnames):
    filename = os.path.join(fp, 'discrete_gridworld.csv')
    with open(filename, 'w', newline='\n') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter=';')

        writer.writeheader()

        for summary in results:
            summary = summary.copy()
            del summary['summaries']

            writer.writerow(summary)


def main(fp_data=None, show_plot=True, save=True, fp=None):
    if fp_data is None:
        fp_data = '/home/bart/Projects/smartstart/thesis/data/discrete_gridworld/29112017/'

    if fp is None:
        fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

    envs = ['GridWorldEasy', 'GridWorldMedium', 'GridWorldHard', 'GridWorldExtreme']
    baselines = [1/31, 1/72, 1/138, 1/223]
    xlims = [[0, int(15e4)], [0, int(2e6)], None, [0, int(8e6)]]
    algos = ['QLearning']

    results = []
    for env, baseline, xlim in zip(envs, baselines, xlims):
        for algo in algos:
            exploration_strategies = [QLearning.BOLTZMANN, QLearning.E_GREEDY, QLearning.COUNT_BASED, QLearning.UCB1]

            summaries = get_summaries(fp_data, env, algo, exploration_strategies)

            plot_summaries(summaries, len(exploration_strategies), baseline, xlim)
            get_rise_time(summaries, baseline, 0.95)
            results += summaries

    for result in results:
        print("%s %s %s %s: %.0f" % (result['env'], result['algo'], result['exploration_strategy'], ("smart start" if result['smart_start'] else ""), result['rise time']))

    if save:
        header = ['env', 'algo', 'exploration_strategy', 'smart_start', 'rise time', 'mean', 'std', 'max']
        save_results(fp, results, header)

    if show_plot:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_plot', action='store_false')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--fp')

    args = parser.parse_args()

    main(show_plot=args.no_plot, save=args.save, fp=args.fp)