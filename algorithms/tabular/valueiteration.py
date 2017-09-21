import random
from collections import defaultdict

import numpy as np


class ValueIteration(object):

    def __init__(self,
                 env,
                 gamma=0.99,
                 min_error=1e-5,
                 max_itr=1000):
        self.env = env
        self.gamma = gamma
        self.min_error = min_error
        self.max_itr = max_itr

        # self.V = np.zeros((self.env.w, self.env.h))
        self.V = defaultdict(lambda: 0)
        self.T = defaultdict(lambda: defaultdict(lambda: 0))
        self.R = defaultdict(lambda: defaultdict(lambda: 0))
        self.obses = set()
        self.goal = None

    def reset(self):
        self.V.clear()
        self.T.clear()
        self.R.clear()
        self.goal = None

    def add_obs(self, obs):
        self.obses.add(tuple(obs))

    def set_goal(self, obs):
        self.add_obs(obs)
        self.goal = tuple(obs)
        self.V[self.goal] = 0

    def set_transition(self, obs, action, obs_tp1, value):
        self.T[tuple(obs) + (action,)][tuple(obs_tp1)] = value

    def get_transition(self, obs, action, obs_tp1=None):
        if obs_tp1 is not None:
            return self.T[tuple(obs) + (action,)][tuple(obs_tp1)]

        return self.T[tuple(obs) + (action,)]

    def set_reward(self, obs, action, obs_tp1, value):
        self.R[tuple(obs) + (action,)][tuple(obs_tp1)] = value

    def optimize(self):
        delta = 0
        for itr in range(self.max_itr):
            delta = 0
            for obs in self.obses:
                if obs != self.goal:
                    v = self.V[obs]
                    v_new = []
                    for action in self.env.possible_actions(obs):
                        v_new.append(self._get_value(obs, action))
                    self.V[obs] = max(v_new)
                    delta = max(delta, abs(v - self.V[obs]))
            # print("Iteration %d finished. Delta: %.2f" % (itr, delta))
            if delta < self.min_error:
                # print("Optimization converged in %d iterations. Final delta: %.2f" % (itr, delta))
                return
        # print("Optimization did not converge in %d iterations. Final delta: %.2f" % (self.max_itr, delta))

    def get_action(self, obs):
        actions = self.env.possible_actions(obs)
        max_actions = []
        max_value = -float('inf')
        for action in actions:
            value = self._get_value(obs, action)
            if value > max_value:
                max_value = value
                max_actions = [action]
            elif value == max_value:
                max_actions.append(action)

        return random.choice(max_actions)

    def _get_value(self, obs, action):
        if tuple(obs) == self.goal:
            return self.V[self.goal]
        value = 0.
        for obs_prime, transition in self.get_transition(obs, action).items():
            r = self.R[tuple(obs) + (action,)][tuple(obs_prime)]
            value += transition * (r + self.gamma * self.V[obs_prime])
        return value

    def get_value_map(self):
        V = np.zeros((self.env.w, self.env.h))
        for i in range(self.env.w):
            for j in range(self.env.h):
                V[i, j] = self.V[(i, j)]
        return V


if __name__ == "__main__":
    import time
    from environments.tabular.gridworld import GridWorld, GridWorldVisualizer

    visualizer = GridWorldVisualizer()
    visualizer.add_visualizer(GridWorldVisualizer.LIVE_AGENT,
                              GridWorldVisualizer.VALUE_FUNCTION,
                              GridWorldVisualizer.DENSITY,
                              GridWorldVisualizer.CONSOLE)

    env = GridWorld.generate(GridWorld.IMPOSSIBRUUHHH)
    env.visualizer = visualizer
    # env.T_prob = 0.1
    env.reset()

    algo = ValueIteration(env, max_itr=10000, min_error=1e-5)
    algo.T, algo.R = env.get_T_R()
    algo.obses = env.get_all_states()
    algo.set_goal(env.goal_state)

    algo.optimize()

    density_map = np.zeros((env.w, env.h))

    obs = env.reset()
    density_map[tuple(obs)] += 1
    env.render(value_map=algo.get_value_map(), density_map=density_map)
    steps = 0
    while True:
        obs, reward, done, _ = env.step(algo.get_action(obs))
        steps += 1
        density_map[tuple(obs)] += 1
        env.render(value_map=algo.get_value_map(), density_map=density_map)

        time.sleep(0.01)

        if done:
            break

    print("Optimal path for %s is %d steps in length" % (env.name, steps))

    render = True
    while render:
        render = env.render(value_map=algo.get_value_map(), density_map=density_map)
