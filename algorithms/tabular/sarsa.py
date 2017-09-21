import numpy as np

from algorithms.tdlearning import TDLearning, TDLearningLambda


class SARSA(TDLearning):

    def __init__(self, env, *args, **kwargs):
        super(SARSA, self).__init__(env, *args, **kwargs)

    def get_next_q_action(self, obs_tp1, done):
        if not done:
            action_tp1 = self.get_action(obs_tp1)
            next_q_value = self.get_q_value(obs_tp1, action_tp1)
        else:
            next_q_value = 0.
            action_tp1 = None

        return next_q_value, action_tp1


class SARSALamba(TDLearningLambda):

    def __init__(self, env, *args, **kwargs):
        super(SARSALamba, self).__init__(env, *args, **kwargs)

    def get_next_q_action(self, obs_tp1, done):
        if not done:
            action_tp1 = self.get_action(obs_tp1)
            next_q_value = self.get_q_value(obs_tp1, action_tp1)
        else:
            next_q_value = 0.
            action_tp1 = None

        return next_q_value, action_tp1


if __name__ == "__main__":
    from environments.tabular.gridworld import GridWorld, GridWorldVisualizer

    directory = '/home/bartkeulen/repositories/smartstart/data/tmp'

    np.random.seed()

    visualizer = GridWorldVisualizer()
    visualizer.add_visualizer(GridWorldVisualizer.LIVE_AGENT,
                              GridWorldVisualizer.CONSOLE,
                              GridWorldVisualizer.VALUE_FUNCTION,
                              GridWorldVisualizer.DENSITY)
    env = GridWorld.generate(GridWorld.EXTREME)
    env.visualizer = visualizer
    # env.wall_reset = True

    agent = SARSALamba(env, alpha=0.3, num_episodes=1000, max_steps=10000)

    summary = agent.train(render=False, render_episode=True)

    summary.save(directory=directory)

