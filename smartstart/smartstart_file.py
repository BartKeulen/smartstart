from collections import defaultdict

import numpy as np

from smartstart import PolicyMap


class SmartStart(object):

    def __init__(self,
                 env,
                 exploration_steps=50,
                 alpha=1.,
                 beta=1.):
        self.env = env
        self.exploration_steps = exploration_steps
        self.alpha = alpha
        self.beta = beta
        self.policy_map = PolicyMap(self.env.reset())

    def get_policy(self, state):
        node = self.policy_map.get_node(state)
        return node.get_policy()

    def get_start(self, algorithm):
        density_map = algorithm.get_density_map()
        if density_map is None:
            return None
        possible_starts = np.asarray(np.where(density_map > 0))
        if not possible_starts.any():
            return None

        smart_start = None
        max_ucb = -float('inf')
        for i in range(possible_starts.shape[1]):
            obs = possible_starts[:, i]
            q_values, _ = algorithm.get_q_values(obs)
            q_value = max(q_values)
            ucb = self.alpha * q_value + self.beta * (1 - density_map[tuple(obs)])
            if ucb > max_ucb:
                smart_start = obs
                max_ucb = ucb
        return smart_start, self.get_policy(smart_start)


if __name__ == "__main__":
    from environments.gridworld import GridWorld, GridWorldVisualizer
    from algorithms.qlearning import QLearning
    from algorithms.sarsa import SARSA
    from algorithms.sarsalambda import SARSALambda
    from algorithms.qlearninglambda import QLearningLambda
    from utilities.scheduler import LinearScheduler

    directory = '/home/bartkeulen/repositories/smartstart/data/tmp'

    np.random.seed()

    visualizer = GridWorldVisualizer()
    visualizer.add_visualizer(GridWorldVisualizer.LIVE_AGENT,
                              GridWorldVisualizer.CONSOLE,
                              GridWorldVisualizer.VALUE_FUNCTION,
                              GridWorldVisualizer.DENSITY)
    env = GridWorld.generate(GridWorld.HARD)
    env.visualizer = visualizer
    env.wall_reset = True

    smart_start = SmartStart(env, exploration_steps=1000, alpha=0.)
    ss_scheduler = LinearScheduler(1500, 2000)

    agent = SARSALambda(env,
                        epsilon=0.,
                          alpha=0.3,
                          num_episodes=10000,
                          max_steps=10000,
                          smart_start=smart_start,
                          ss_scheduler=ss_scheduler)

    summary = agent.train(render_episode=True)

    summary.save(directory=directory)

    while True:
        pass
