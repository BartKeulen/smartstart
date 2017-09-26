import numpy as np

from smartstart.algorithms import ValueIteration
from smartstart.environments.gridworld import GridWorld, GridWorldVisualizer

gridworlds = [GridWorld.EASY, GridWorld.MEDIUM, GridWorld.HARD, GridWorld.EXTREME]

"""
OPTIMAL PATHS:

EASY:       31
MEDIUM:     72
HARD:       138
EXTREME:    223
"""

for gridworld in gridworlds:
    env = GridWorld.generate(gridworld)
    visualizer = GridWorldVisualizer(env)
    visualizer.add_visualizer(GridWorldVisualizer.LIVE_AGENT,
                              GridWorldVisualizer.VALUE_FUNCTION,
                              GridWorldVisualizer.DENSITY,
                              GridWorldVisualizer.CONSOLE)

    env.visualizer = visualizer
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

        if done:
            break

    print("Optimal path for %s is %d steps in length" % (env.name, steps))

    render = True
    while render:
        render = env.render(value_map=algo.get_value_map(), density_map=density_map)
