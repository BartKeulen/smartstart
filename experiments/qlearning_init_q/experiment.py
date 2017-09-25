import numpy as np
from utilities.experimenter import run_experiment

from smartexploration.algorithms import QLearning
from smartexploration.algorithms import TDLearning
from smartexploration.environments.gridworld import GridWorld
from smartexploration.utilities.utilities import get_data_directory

directory = get_data_directory(__file__)

init_q_values = [0., 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.]
exploration_strategies = [TDLearning.E_GREEDY, TDLearning.BOLTZMANN, TDLearning.NONE]
num_exp = 25


def task(params):
    np.random.seed()

    num_episodes = 1000
    max_steps = 1000

    env = GridWorld.generate(GridWorld.EASY)

    agent = QLearning(env,
                      alpha=0.1,
                      gamma=0.99,
                      num_episodes=num_episodes,
                      max_steps=max_steps,
                      exploration=params['exploration_strategy'],
                      init_q_value=params['init_q_value'],
                      epsilon=0.05)

    post_fix = "exploration=%s_init_q=%.7f_%d" % (params['exploration_strategy'], params['init_q_value'], params['run'])

    summary = agent.train(render=False, render_episode=False, print_results=False)

    summary.save_to_gcloud(directory='qlearning_init_q', post_fix=post_fix)


param_grid = {'task': task,
              'init_q_value': init_q_values,
              'exploration_strategy': exploration_strategies,
              'num_exp': num_exp}


if __name__ == "__main__":
    run_experiment(param_grid)