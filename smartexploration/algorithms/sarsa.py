"""SARSA

Module defining classes for SARSA and SARSA(lambda).

See 'Reinforcement Learning: An Introduction by Richard S. Sutton and
Andrew G. Barto for more information.
"""
from smartexploration.algorithms import TDLearning, TDLearningLambda


class SARSA(TDLearning):
    """SARSA

    """

    def __init__(self, env, *args, **kwargs):
        """Constructs SARSA object

        Args:
            env:        environment
            *args:      see parent :class:`class TDLearning
            <smartexploration.TDLearning>'
            **kwargs:   see parent :class:`class TDLearning
            <smartexploration.TDLearning>'
        """
        super(SARSA, self).__init__(env, *args, **kwargs)

    def get_next_q_action(self, obs_tp1, done):
        """On-policy action selection

        Args:
            obs_tp1:    Next observation
            done:       Boolean is True for terminal state

        Returns:
            Q-value for obs_tp1 and accompanying action_tp1
        """
        if not done:
            action_tp1 = self.get_action(obs_tp1)
            next_q_value = self.get_q_value(obs_tp1, action_tp1)
        else:
            next_q_value = 0.
            action_tp1 = None

        return next_q_value, action_tp1


class SARSALambda(TDLearningLambda):
    """SARSA(lambda)

    """

    def __init__(self, env, *args, **kwargs):
        """Constructs SARSALambda object

        Args:
            env:        environment
            *args:      see parent :class:`class TDLearning
            <smartexploration.TDLearningLambda>'
            **kwargs:   see parent :class:`class TDLearning
            <smartexploration.TDLearningLambda>'
        """
        super(SARSALambda, self).__init__(env, *args, **kwargs)

    def get_next_q_action(self, obs_tp1, done):
        """On-policy action selection

        Args:
            obs_tp1:    Next observation
            done:       Boolean is True for terminal state

        Returns:
            Q-value for obs_tp1 and accompanying action_tp1
        """
        if not done:
            action_tp1 = self.get_action(obs_tp1)
            next_q_value = self.get_q_value(obs_tp1, action_tp1)
        else:
            next_q_value = 0.
            action_tp1 = None

        return next_q_value, action_tp1

#
# if __name__ == "__main__":
#     from smartexploration.environments.gridworld import GridWorld, GridWorldVisualizer
#
#     directory = '/home/bartkeulen/repositories/smartexploration/data/tmp'
#
#     np.random.seed()
#
#     env = GridWorld.generate(GridWorld.EXTREME)
#     visualizer = GridWorldVisualizer(env)
#     visualizer.add_visualizer(GridWorldVisualizer.LIVE_AGENT,
#                               GridWorldVisualizer.CONSOLE,
#                               GridWorldVisualizer.VALUE_FUNCTION,
#                               GridWorldVisualizer.DENSITY)
#     env.visualizer = visualizer
#     # env.wall_reset = True
#
#     agent = SARSALambda(env, alpha=0.3, num_episodes=1000, max_steps=10000)
#
#     summary = agent.train(render=False, render_episode=True)
#
#     summary.save(directory=directory)

