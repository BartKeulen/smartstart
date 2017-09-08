import numpy as np

from algorithms.qlearning import QLearning
from smartstart.smartstart_file import SmartStart


class QLearningLambda(QLearning):

    def __init__(self, env, lamb=0.75, threshold_traces=1e-3, *args, **kwargs):
        super(QLearningLambda, self).__init__(env, *args, **kwargs)
        self.lamb = lamb
        self.threshold_traces = threshold_traces
        self.traces = np.zeros((self.env.w, self.env.h, self.env.num_actions))

    def update_q_value(self, obs, action, td_error):
        idx = tuple(obs) + (action,)
        self.traces[idx] = 1
        active_traces = np.asarray(np.where(self.traces > self.threshold_traces))
        for i in range(active_traces.shape[1]):
            idx = tuple(active_traces[:, i])
            self.Q[idx] += self.alpha * td_error * self.traces[idx]
            self.traces[idx] *= self.gamma * self.lamb


if __name__ == "__main__":
    from environments.gridworld import GridWorld, GridWorldVisualizer

    directory = '/home/bartkeulen/repositories/smartstart/data/tmp'

    np.random.seed()

    visualizer = GridWorldVisualizer()
    visualizer.add_visualizer(GridWorldVisualizer.LIVE_AGENT,
                              GridWorldVisualizer.CONSOLE,
                              GridWorldVisualizer.VALUE_FUNCTION,
                              GridWorldVisualizer.DENSITY)
    env = GridWorld.generate(GridWorld.EASY)
    env.visualizer = visualizer
    env.wall_reset = True

    smart_start = SmartStart(env, alpha=0., beta=1.)
    agent = QLearningLambda(env, num_episodes=1000, max_steps=1000, smart_start=smart_start)
    # agent = QLearningLambda(env, num_episodes=1000, max_steps=25000)

    summary = agent.train()

    summary.save(directory=directory)