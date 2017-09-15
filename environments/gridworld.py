import math
import time

import numpy as np

from environments.gridworldvisualizer import GridWorldVisualizer
from environments.presetgridworlds import *
from environments.generate_maze import generate_maze


class GridWorld(object):
    EASY = 0
    MEDIUM = 1
    HARD = 2
    EXTREME = 3
    IMPOSSIBRUUHHH = 4

    def __init__(self, name, layout, T_prob=0.1, wall_reset=False, visualizer=None, scale=5):
        self.name = name
        layout = np.asarray(layout)
        self.T_prob = T_prob

        grid_world = np.kron(layout, np.ones((scale, scale), dtype=layout.dtype))
        start_state = np.asarray(np.where(grid_world == 2))[:, math.floor(scale**2/2)]
        goal_state = np.asarray(np.where(grid_world == 3))[:, math.floor(scale**2/2)]
        grid_world[grid_world == 2] = 0
        grid_world[grid_world == 3] = 0
        grid_world[tuple(start_state)] = 2
        grid_world[tuple(goal_state)] = 3

        self.w, self.h = grid_world.shape
        self.num_actions = 4
        self.grid_world, self.start_state, self.goal_state = grid_world, start_state, goal_state

        self.wall_reset = wall_reset

        if visualizer is None:
            self.visualizer = GridWorldVisualizer(self.name)
            self.visualizer.add_visualizer(GridWorldVisualizer.LIVE_AGENT)
        else:
            self.visualizer = visualizer

    @property
    def visualizer(self):
        return self._visualizer

    @visualizer.setter
    def visualizer(self, visualizer):
        self._visualizer = visualizer
        self._visualizer.name = self.name

    def reset(self, start_state=None):
        if start_state is not None:
            self.state = start_state
        else:
            self.state = self.start_state.copy()
        return self.start_state

    def _move(self, action):
        if action == 0:
            movement = np.array([0, 1])
        elif action == 1:
            movement = np.array([1, 0])
        elif action == 2:
            movement = np.array([0, -1])
        elif action == 3:
            movement = np.array([-1, 0])
        else:
            raise NotImplementedError("Invalid action %d. Available actions: [0, 1, 2, 3]")

        return self.state + movement

    def step(self, action):
        # if np.random.rand() < self.T_prob:
        #     action = np.random.choice(self.possible_actions(self.state))

        new_state = self._move(action)

        if (new_state < 0).any() or (new_state[0] >= self.w) or (new_state[1] >= self.h) or (self.grid_world[tuple(new_state)] == 1):
            if self.wall_reset:
                return self.state, 0., True, {}
            else:
                new_state = self.state.copy()

        r = 1. if np.array_equal(new_state, self.goal_state) else 0.
        done = True if np.array_equal(new_state, self.goal_state) else False

        self.state = new_state

        return self.state, r, done, {}

    def possible_actions(self, state):
        return [0, 1, 2, 3]

    def _get_grid(self):
        grid_copy = self.grid_world.copy()
        grid_copy[tuple(self.state)] = 4
        return grid_copy

    def close_render(self):
        return self.visualizer.render(self._get_grid(), close=True)

    def render(self, **kwargs):
        if self._visualizer is None:
            self._visualizer = GridWorldVisualizer()
            self._visualizer.add_visualizer(GridWorldVisualizer.LIVE_AGENT,
                                      GridWorldVisualizer.VALUE_FUNCTION,
                                      GridWorldVisualizer.DENSITY,
                                      GridWorldVisualizer.CONSOLE)
        return self.visualizer.render(self._get_grid(), **kwargs)

    @staticmethod
    def generate(type=EASY, size=None):
        if type == GridWorld.EASY:
            name, layout, scale = gridworld_easy()
        elif type == GridWorld.MEDIUM:
            name, layout, scale = gridworld_medium()
        elif type == GridWorld.HARD:
            name, layout, scale = gridworld_hard()
        elif type == GridWorld.EXTREME:
            name, layout, scale = gridworld_extreme()
        elif type == GridWorld.IMPOSSIBRUUHHH:
            name, layout, scale = generate_maze(size=size)
        else:
            raise NotImplementedError("Please choose from the available GridWorld implementations or build one your self.")

        return GridWorld(name, layout, scale=scale)


if __name__ == "__main__":
    env = GridWorld.generate(GridWorld.EASY)
    env.wall_reset = True

    obs = env.reset()

    steps = []
    num_episodes = 50
    reached_goal = False
    i_episode = 0
    while not reached_goal:
        obs = env.reset()
        start = time.time()
        i_step = 0
        while True:
            action = np.random.randint(4)

            # env.render()

            obs_tp1, r, done, _ = env.step(action)

            obs = obs_tp1
            i_step += 1

            # if i_step % 10000 == 0:
            #     elapsed = time.time() - start
            #     print("%d steps in %.2f seconds" % (i_step, elapsed))

            if done:
                if r > 0:
                    reached_goal = True
                break

        elapsed = time.time() - start
        print("Episode %d finished in %d steps and %.2f seconds" % (i_episode, i_step, elapsed))
        steps.append(i_step)
        i_episode += 1

    print("")
    print("GOAL REACHED IN %d EPISODES" % i_episode)
    # print("Average number of steps over %d episodes is %.f" % (num_episodes, sum(steps) / len(steps)))