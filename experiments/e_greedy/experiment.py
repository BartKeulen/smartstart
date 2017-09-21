import numpy as np
from algorithms.qlearning import QLearning
from algorithms.tdlearning import TDLearning

from environments.tabular.gridworld import GridWorld
from utilities.experimenter import run_experiment
from utilities.utilities import get_data_directory

directory = get_data_directory(__file__)

maze_type = [GridWorld.EASY, GridWorld.MEDIUM]
alphas = [0.01, 0.05, 0.1, 0.25, 0.5, 1.]
epsilons = [0.01, 0.05, 0.1, 0.25]
gammas = [0.5, 0.9, 0.95, 0.99, 0.999]
num_exp = 1


def task(params):
    np.random.seed()

    if params['maze_type'] == GridWorld.EASY:
        num_episodes = 1000
        max_steps = 1000
    elif params['maze_type'] == GridWorld.MEDIUM:
        num_episodes = 2500
        max_steps = 2500
    elif params['maze_type'] == GridWorld.HARD:
        num_episodes = 10000
        max_steps = 10000
    else:
        raise Exception("Not a valid maze type for this experiment")

    env = GridWorld.generate(params['maze_type'])

    agent = QLearning(env,
                      alpha=params['alpha'],
                      gamma=params['gamma'],
                      num_episodes=num_episodes,
                      max_steps=max_steps,
                      exploration=TDLearning.E_GREEDY,
                      epsilon=params['epsilon'])

    post_fix = "alpha=%.2f_gamma=%.3f_epsilon=%.2f_%d" % (params['alpha'], params['gamma'], params['epsilon'], params['run'])

    summary = agent.train(render=False, render_episode=False, print_results=False)

    summary.save(directory=directory, post_fix=post_fix)


param_grid = {'task': task,
              'maze_type': maze_type,
              'alpha': alphas,
              'epsilon': epsilons,
              'gamma': gammas,
              'num_exp': num_exp}


if __name__ == "__main__":
    run_experiment(param_grid)