import argparse
import csv
import os
import ast

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set('paper')


def get_data(fp):
    with open(fp, 'r', newline='\n') as csvfile:
        fieldnames = ['env', 'exploration_strategy', 'smart_start', 'num_iter', 'num_episodes', 'max_steps', 'mean', 'std', 'steps']
        reader = csv.DictReader(csvfile, fieldnames=fieldnames, delimiter=';')

        data = [row for row in reader]
        data.pop(0)
    return data


def plot_subplot(ax, data, env, exploration_strategy):
    x = []
    y = []
    std = []
    x_ss = []
    y_ss = []
    std_ss = []
    for row in data:
        if row['env'] == env:
            if row['exploration_strategy'] == exploration_strategy:
                if not eval(row['smart_start']):
                    x.append(int(row['max_steps']))
                    y.append(float(row['mean']))
                    std.append(float(row['std']))
                else:
                    x_ss.append(int(row['max_steps']))
                    y_ss.append(float(row['mean']))
                    std_ss.append(float(row['std']))

    linestyle = {"linestyle": 'None', "markeredgewidth": 1, "elinewidth": 0.5, "capsize": 10}
    # if x:
    #     ax.errorbar(x, y, yerr=std, marker='o', color="b", label='Normal', **linestyle)
    if x_ss:
        ax.errorbar(x_ss, y_ss, yerr=std_ss, marker='^', color="g", label='With Smart Start', **linestyle)

    ax.set_xlim(100, 100000)
    ax.set_xscale('log')


def plot_figure(data, envs, expls):
    fig = plt.figure(figsize=(12, 10))
    idx = 1
    for expl in expls:
        for env in envs:
            ax = fig.add_subplot(len(expls), len(envs), idx)
            plot_subplot(ax, data, env, expl)
            idx += 1

    ax = fig.get_axes()[0]
    # ax.legend((ax.lines[0], ax.lines[3]), ('Normal', 'Smart Start'), loc='upper left')

    fig.text(0.5, 0.03, 'Maximum Steps per Episode', ha='center')
    fig.text(0.3, 0.06, envs[0], va='center')
    fig.text(0.7, 0.06, envs[1], va='center')

    fig.text(0.03, 0.5, r'Average Steps until Goal ($\mu \pm \sigma$)', va='center', rotation='vertical')
    fig.text(0.06, 0.75, 'Random', ha='center', rotation='vertical')
    fig.text(0.06, 0.50, 'Count-Based', ha='center', rotation='vertical')
    fig.text(0.06, 0.25, 'Model-Based', ha='center', rotation='vertical')

    return fig


def save_fig(fp, fig, envs):
    fig.savefig(os.path.join(fp, 'exploration_%s-%s.pdf' % tuple(envs)),
                format='pdf',
                bbox_inches='tight')


def main(fp_csvfile=None, show_plot=False, save=False, fp=None):
    if fp_csvfile is None:
        fp_csvfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/exploration_25112017.csv')

    if fp is None:
        fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img')

    data = get_data(fp_csvfile)
    exploration_strategies = ['random', 'count-based', 'model-based']

    envs_1 = ['Easy', 'Hard']
    fig_1 = plot_figure(data, envs_1, exploration_strategies)

    envs_2 = ['Medium', 'Extreme']
    fig_2 = plot_figure(data, envs_2, exploration_strategies)

    if save:
        save_fig(fp, fig_1, envs_1)
        save_fig(fp, fig_2, envs_2)

    if show_plot:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_plot', action='store_false')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--fp_csv')
    parser.add_argument('--fp')

    args = parser.parse_args()

    main(args.fp_csv, args.no_plot, args.save, args.fp)
