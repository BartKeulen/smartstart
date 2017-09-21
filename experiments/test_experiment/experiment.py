import numpy as np
from algorithms.qlearning import QLearning
from algorithms.tdlearning import TDLearning

from environments.tabular.gridworld import GridWorld
from utilities.experimenter import run_experiment
from utilities.utilities import get_data_directory

directory = get_data_directory(__file__)

alphas = [0.05, 0.1]
num_exp = 5


def task(params):
    np.random.seed()

    num_episodes = 25
    max_steps = 100

    env = GridWorld.generate(GridWorld.EASY)

    agent = QLearning(env,
                      alpha=params['alpha'],
                      gamma=0.99,
                      num_episodes=num_episodes,
                      max_steps=max_steps,
                      exploration=TDLearning.E_GREEDY,
                      epsilon=0.05)

    post_fix = "alpha=%.2f_%d" % (params['alpha'], params['run'])

    summary = agent.train(render=False, render_episode=False, print_results=False)

    summary.save_to_gcloud(directory='test', post_fix=post_fix)


param_grid = {'task': task,
              'alpha': alphas,
              'num_exp': num_exp}


if __name__ == "__main__":
    run_experiment(param_grid)