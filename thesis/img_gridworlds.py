import os
import argparse

import pygame

from smartstart.environments import GridWorld
from smartstart.environments import GridWorldVisualizer


def main(show_plot=False, save=True, fp=None):
    if fp is None:
        fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img')

    for env_name in [GridWorld.EASY, GridWorld.MEDIUM, GridWorld.HARD, GridWorld.EXTREME]:
        env = GridWorld.generate(env_name)
        vis = GridWorldVisualizer(env)
        vis.add_visualizer(GridWorldVisualizer.LIVE_AGENT)
        vis.colors.update({
            'background': (255, 255, 255, 255),
            'wall': (0, 0, 0, 0),
            'agent': (255, 0, 0, 255)
        })

        print('%s, start state: %s, goal state: %s' % (env.name, env.start_state, env.goal_state))

        env.reset()
        env.render()

        directory = os.path.join(fp, '%s.png' % env.name)

        if save:
            pygame.image.save(vis.screen, directory)

        while True:
            render = env.render(close=(not show_plot))
            if not render:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--show_plot', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--fp')

    args = parser.parse_args()

    main(show_plot=args.show_plot, save=args.save, fp=args.fp)
