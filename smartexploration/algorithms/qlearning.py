"""Q-Learning

Module defining classes for Q-Learning and Q(lambda).

See 'Reinforcement Learning: An Introduction by Richard S. Sutton and
Andrew G. Barto for more information.
"""
from smartexploration.algorithms import TDLearning, TDLearningLambda


class QLearning(TDLearning):
    """Q-Learning

    """

    def __init__(self, env, *args, **kwargs):
        """Constructs QLearning object

        Args:
            env:        environment
            *args:      see parent :class:`class TDLearning
            <smartexploration.TDLearning>'
            **kwargs:   see parent :class:`class TDLearning
            <smartexploration.TDLearning>'
        """
        super(QLearning, self).__init__(env, *args, **kwargs)

    def get_next_q_action(self, obs_tp1, done):
        """Off-policy action selection

        Args:
            obs_tp1:    Next observation
            done:       Boolean is True for terminal state

        Returns:
            Q-value for obs_tp1 and accompanying action_tp1
        """
        if not done:
            next_q_values, _ = self.get_q_values(obs_tp1)
            next_q_value = max(next_q_values)
            action_tp1 = self.get_action(obs_tp1)
        else:
            next_q_value = 0.
            action_tp1 = None

        return next_q_value, action_tp1


class QLearningLambda(TDLearningLambda):
    """Q(lambda)

    Note:
        Does not work properly, because q-learning is off-policy standard
        eligibility traces might fail.
    """

    def __init__(self, env, *args, **kwargs):
        """Constructs QLearningLambda object

        Args:
            env:        environment
            *args:      see parent :class:`class TDLearning
            <smartexploration.TDLearningLambda>'
            **kwargs:   see parent :class:`class TDLearning
            <smartexploration.TDLearningLambda>'
        """
        super(QLearningLambda, self).__init__(env, *args, **kwargs)

    def get_next_q_action(self, obs_tp1, done):
        """Off-policy action selection

        Args:
            obs_tp1:    Next observation
            done:       Boolean is True for terminal state

        Returns:
            Q-value for obs_tp1 and accompanying action_tp1
        """
        if not done:
            next_q_values, _ = self.get_q_values(obs_tp1)
            next_q_value = max(next_q_values)
            action_tp1 = self.get_action(obs_tp1)
        else:
            next_q_value = 0.
            action_tp1 = None

        return next_q_value, action_tp1

#
# if __name__ == "__main__":
#     from smartexploration.environments.gridworld import GridWorld, GridWorldVisualizer
#     import time
#
#     start = time.time()
#
#     directory = '/home/bartkeulen/repositories/smartexploration/data/tmp'
#
#     np.random.seed()
#
#     grid_world = GridWorld.generate(GridWorld.EASY)
#     visualizer = GridWorldVisualizer(grid_world)
#     visualizer.add_visualizer(GridWorldVisualizer.LIVE_AGENT,
#                               GridWorldVisualizer.CONSOLE,
#                               GridWorldVisualizer.VALUE_FUNCTION,
#                               GridWorldVisualizer.DENSITY)
#     grid_world.visualizer = visualizer
#     # env.wall_reset = True
#
#     agent = QLearning(grid_world, alpha=0.3, num_episodes=10000, max_steps=1000,
#                       exploration=QLearning.E_GREEDY)
#
#     summary = agent.train(render=False, render_episode=True,
#                           print_results=False)
#
#     summary.save(directory=directory)
#
#     print("Time elapsed: %.0f seconds" % (time.time() - start))


