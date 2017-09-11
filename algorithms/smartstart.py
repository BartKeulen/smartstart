import numpy as np

from algorithms.policy import PolicyMap
from utilities.datacontainers import Episode, SummarySmall, Summary


def SmartStart(base, env, *args, **kwargs):

    class _SmartStart(base):

        def __init__(self,
                     env,
                     exploration_steps=50,
                     alpha=1.,
                     beta=1.,
                     eta=0.5,
                     *args,
                     **kwargs):
            super(_SmartStart, self).__init__(env, *args, **kwargs)
            self.__class__.__name__ = "SmartStart_" + base.__name__
            self.exploration_steps = exploration_steps
            self.alpha = alpha
            self.beta = beta
            self.eta = eta
            self.policy_map = PolicyMap(self.env.reset())

        def get_start(self):
            density_map = self.get_density_map()
            if density_map is None:
                return None
            possible_starts = np.asarray(np.where(density_map > 0))
            if not possible_starts.any():
                return None

            smart_start = None
            max_ucb = -float('inf')
            for i in range(possible_starts.shape[1]):
                obs = possible_starts[:, i]
                q_values, _ = self.get_q_values(obs)
                q_value = max(q_values)
                ucb = self.alpha * q_value + self.beta * (1 - density_map[tuple(obs)])
                if ucb > max_ucb:
                    smart_start = obs
                    max_ucb = ucb
            return smart_start, self._get_policy(smart_start)

        def _get_policy(self, state):
            node = self.policy_map.get_node(state)
            return node.get_policy()

        def train(self, summary_to_use=SummarySmall, render=False, render_episode=False, print_results=True):
            summary = summary_to_use(self.__class__.__name__ + "_" + self.env.name)

            for i_episode in range(self.num_episodes):
                episode = Episode()

                obs = self.env.reset()
                self.policy_map.reset_to_root()

                max_steps = self.max_steps
                if i_episode > 0 and np.random.rand() <= self.eta:
                    start_state, policy = self.get_start()

                    for action in policy:
                        obs, _, done, render = self.take_step(obs, action, episode, render)

                        if done:
                            break

                    max_steps = self.exploration_steps

                action = self.get_action(obs)

                for step in range(max_steps):
                    obs, action, done, render = self.take_step(obs, action, episode, render)

                    if done:
                        break

                # Render and/or print results
                message = "Episode: %d, steps: %d, reward: %.2f" % (i_episode, len(episode), episode.total_reward())
                if render or render_episode:
                    value_map = self.Q.copy()
                    value_map = np.max(value_map, axis=2)
                    render_episode = self.env.render(value_map=value_map, density_map=self.get_density_map(),
                                                     message=message)
                if print_results:
                    print("Episode: %d, steps: %d, reward: %.2f" % (i_episode, len(episode), episode.total_reward()))
                summary.append(episode)

            while render:
                value_map = self.Q.copy()
                value_map = np.max(value_map, axis=2)
                render = self.env.render(value_map=value_map, density_map=self.get_density_map())

            return summary

        def take_step(self, obs, action, episode, render=False):
            obs_tp1, r, done, _ = self.env.step(action)

            if render:
                value_map = self.Q.copy()
                value_map = np.max(value_map, axis=2)
                render = self.env.render(value_map=value_map, density_map=self.get_density_map())

            _, action_tp1 = self.update_q_value(obs, action, r, obs_tp1, done)

            self.increment(obs)
            self.policy_map.add_node(obs_tp1, action)

            episode.append(obs, action, r, obs_tp1, done)

            return obs_tp1, action_tp1, done, render

    return _SmartStart(env, *args, **kwargs)


if __name__ == "__main__":
    from environments.gridworld import GridWorld, GridWorldVisualizer
    from algorithms.qlearning import QLearning, QLearningLambda
    from algorithms.sarsa import SARSA, SARSALamba

    directory = '/home/bartkeulen/repositories/smartstart/data/tmp'

    np.random.seed()

    visualizer = GridWorldVisualizer()
    visualizer.add_visualizer(GridWorldVisualizer.LIVE_AGENT,
                              GridWorldVisualizer.CONSOLE,
                              GridWorldVisualizer.VALUE_FUNCTION,
                              GridWorldVisualizer.DENSITY)
    env = GridWorld.generate(GridWorld.EASY)
    env.visualizer = visualizer
    # env.wall_reset = True

    agent = SmartStart(QLearning, env, alpha=0.3, num_episodes=1000, max_steps=500)

    summary = agent.train(render=False, render_episode=True)

    summary.save(directory=directory)
