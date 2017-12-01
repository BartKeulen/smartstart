import os
import argparse
from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from smartstart.algorithms import TriangularFeature

sns.set(font_scale=1.5)


def main(show_plot=True, save=True, fp=None):
    if fp is None:
        fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img')

    num_states = 1
    obs_min = -10
    obs_max = -obs_min
    num_features_dim = 5
    N = 1000

    feature = TriangularFeature(num_states, [obs_min], [obs_max], num_features_dim)

    x = 1
    print("Feature for s = %d is: %s" % (x, feature.get([x])))

    x = np.linspace(obs_min, obs_max, N)
    y = np.zeros((N, num_features_dim))

    for i, obs in enumerate(x):
        y[i, :] = feature.get([obs])

    lines = ['-', '--', '-.', ':']
    linecycler = cycle(lines)
    plt.figure(figsize=(12, 5))
    for i in range(num_features_dim):
        plt.plot(x, y[:, i], next(linecycler))
    plt.xlabel(r'$\mathbf{s}$')
    plt.ylabel('Activation')
    plt.legend([r'$\phi_%d(\mathbf{s})$' % i for i in range(num_features_dim)],
               loc=2,
               bbox_to_anchor=(1.05, 1),
               borderaxespad=0.)
    plt.autoscale(enable=True, tight=True)

    if save:
        plt.savefig(os.path.join(fp, 'triangular_feature.pdf'),
                    format='pdf',
                    dpi=1200,
                    bbox_inches="tight")

    if show_plot:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_plot', action='store_false')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--fp')

    args = parser.parse_args()

    main(show_plot=args.no_plot, save=args.save, fp=args.fp)

