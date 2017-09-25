import math
from collections import defaultdict

import numpy as np
from smartexploration.environments.generate_gridworld import generate_gridworld
from smartexploration.environments.presets import *

from smartexploration.environments.gridworldvisualizer import GridWorldVisualizer


class GridWorld(object):
    EASY = 0
    MEDIUM = 1
    HARD = 2
    EXTREME = 3
    IMPOSSIBRUUHHH = 4

    def __init__(self, name, layout, T_prob=0., wall_reset=False, visualizer=None, scale=5):
        self.name = self.__class__.__name__ + name
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

        if visualizer is not None:
            self.visualizer = visualizer
        else:
            self._visualizer = None

    def get_all_states(self):
        h, w = self.grid_world.shape
        states = set()
        for y in range(h):
            for x in range(w):
                if self.grid_world[y, x] != 1:
                    states.add((y, x))
        return states

    def get_T_R(self):
        states = self.get_all_states()

        T = defaultdict(lambda: defaultdict(lambda: 0))
        R = defaultdict(lambda: defaultdict(lambda: 0))
        for state in states:
            cur_state = np.asarray(state)
            for action in self.possible_actions(cur_state):
                for p_action in self.possible_actions(cur_state):
                    new_state = self._move(cur_state.copy(), p_action)
                    if (new_state < 0).any() or (new_state[0] >= self.w) or (new_state[1] >= self.h) or (
                        self.grid_world[tuple(new_state)] == 1):
                        new_state = cur_state.copy()

                    if self.grid_world[tuple(new_state)] == 3:
                        R[tuple(cur_state) + (action,)][tuple(new_state)] = 1.

                    if action == p_action:
                        p = 1 - self.T_prob + self.T_prob / len(self.possible_actions(cur_state))
                    else:
                        p = self.T_prob / len(self.possible_actions(cur_state))

                    T[tuple(cur_state) + (action,)][tuple(new_state)] += p

        return T, R

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
        return self.state

    def _move(self, state, action):
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

        return state + movement

    def step(self, action):
        if np.random.rand() < self.T_prob:
            action = np.random.choice(self.possible_actions(self.state))

        new_state = self._move(self.state, action)

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

    def get_grid(self):
        grid_copy = self.grid_world.copy()
        return grid_copy

    def close_render(self):
        return self.visualizer.render(close=True)

    def render(self, **kwargs):
        if self._visualizer is None:
            print("No visualizer attached")
            return False
        return self.visualizer.render(**kwargs)

    @classmethod
    def generate(cls, type=EASY, size=None):
        if type == GridWorld.EASY:
            name, layout, scale = easy()
        elif type == GridWorld.MEDIUM:
            name, layout, scale = medium()
        elif type == GridWorld.HARD:
            name, layout, scale = hard()
        elif type == GridWorld.EXTREME:
            name, layout, scale = extreme()
        elif type == GridWorld.IMPOSSIBRUUHHH:
            name, layout, scale = generate_gridworld(size=size)
        else:
            raise NotImplementedError("Please choose from the available GridWorld implementations or build one your self.")

        return cls(name, layout, scale=scale)


if __name__ == "__main__":
    import time

    env = GridWorld.generate(GridWorld.MEDIUM)
    vis = GridWorldVisualizer(env)
    vis.add_visualizer(GridWorldVisualizer.LIVE_AGENT)
    env.visualizer = vis

    obs = env.reset()

    steps = []
    num_episodes = 50
    reached_goal = False
    i_episode = 0
    render = True
    while not reached_goal:
        obs = env.reset()
        start = time.time()
        i_step = 0
        while True:
            action = np.random.randint(4)

            if render:
                render = env.render()

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
    print("Average number of steps over %d episodes is %.f" % (num_episodes, sum(steps) / len(steps)))