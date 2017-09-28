"""SmartStart module

Defines method for generating a SmartStart object from an algorithm object. The
SmartStart object will be a sublcass of the orignal algorithm object.
"""
import numpy as np

from smartstart.algorithms import ValueIteration
from smartstart.utilities.datacontainers import Episode, Summary


def generate_smartstart_object(base, env, *args, **kwargs):
    """Generates SmartStart object



    Parameters
    ----------
    base : :obj:`~smartstart.algorithms.tdlearning.TDLearning`
        base algorithm to convert to SmartStart algorithm
    env : :obj:`~smartstart.environments.environment.Environment`
        environment
    *args :
        see :obj:`~smarstart.smartexploration.smartexploration.SmartStart`
        and the base class for possible parameters
    **kwargs :
        see :obj:`~smarstart.smartexploration.smartexploration.SmartStart`
        and the base class for possible parameters

    Returns
    -------

    """

    class SmartStart(base):
        """

        """

        def __init__(self,
                     env,
                     exploration_steps=50,
                     exploitation_param=1.,
                     exploration_param=2.,
                     eta=0.5,
                     *args,
                     **kwargs):
            super(SmartStart, self).__init__(env, *args, **kwargs)
            self.__class__.__name__ = "SmartStart_" + base.__name__
            self.exploration_steps = exploration_steps
            self.exploitation_param = exploitation_param
            self.exploration_param = exploration_param
            self.eta = eta
            self.m = 1

            self.policy = ValueIteration(self.env)
            # self.policy_map = PolicyMap(self.env.reset())

        def get_start(self):
            """ """
            count_map = self.get_count_map()
            if count_map is None:
                return None
            possible_starts = np.asarray(np.where(count_map > 0))
            if not possible_starts.any():
                return None

            smart_start = None
            max_ucb = -float('inf')
            for i in range(possible_starts.shape[1]):
                obs = possible_starts[:, i]
                q_values, _ = self.get_q_values(obs)
                q_value = max(q_values)
                ucb = self.exploitation_param * q_value + \
                      np.sqrt((self.exploration_param * np.log(np.sum(count_map)))/count_map[tuple(obs)])
                if ucb > max_ucb:
                    smart_start = obs
                    max_ucb = ucb
            return smart_start

        # def get_start(self):
        #     density_map = self.get_density_map()
        #     if density_map is None:
        #         return None
        #     possible_starts = np.asarray(np.where(density_map > 0))
        #     if not possible_starts.any():
        #         return None
        #
        #     smart_start = None
        #     max_ucb = -float('inf')
        #     for i in range(possible_starts.shape[1]):
        #         obs = possible_starts[:, i]
        #         q_values, _ = self.get_q_values(obs)
        #         q_value = max(q_values)
        #         ucb = self.exploitation_param * q_value + self.exploration_param * (1 - density_map[tuple(obs)])
        #         if ucb > max_ucb:
        #             smart_start = obs
        #             max_ucb = ucb
        #     return smart_start

        # def _get_policy(self, state):
        #     node = self.policy_map.get_node(state)
        #     return node.get_policy()

        def train(self, render=False, render_episode=False, print_results=True):
            """

            Parameters
            ----------
            render :
                 (Default value = False)
            render_episode :
                 (Default value = False)
            print_results :
                 (Default value = True)

            Returns
            -------

            """
            summary = Summary(self.__class__.__name__ + "_" + self.env.name)

            for i_episode in range(self.num_episodes):
                episode = Episode()

                obs = self.env.reset()
                self.policy.add_obs(obs)
                # self.policy_map.reset_to_root()

                max_steps = self.max_steps
                start_state = obs
                if i_episode > 0 and np.random.rand() <= self.eta:
                    start_state = self.get_start()

                    self.policy.reset()
                    finished = False
                    for obs_c, obs_count in self.count_map.items():
                        for action, action_count in obs_count.items():
                            for obs_tp1, count in action_count.items():
                                if obs_tp1 == tuple(start_state):
                                    self.policy.R[obs_c + action][obs_tp1] = 1.
                                self.policy.T[obs_c + action][obs_tp1] = count / sum(self.count_map[obs_c][action].values())

                    self.policy.optimize()

                    for i in range(self.max_steps):
                        action = self.policy.get_action(obs)
                        obs, _, done, render = self.take_step(obs, action, episode, render)

                        if np.array_equal(obs, start_state):
                            break

                        if done:
                            finished = True
                            break

                    max_steps = self.max_steps - len(episode)

                    if finished:
                        max_steps = 0

                if render or render_episode:
                    value_map = self.Q.copy()
                    value_map = np.max(value_map, axis=2)
                    render_episode = self.env.render(value_map=value_map, density_map=self.get_density_map())

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
            """

            Parameters
            ----------
            obs :
                
            action :
                
            episode :
                
            render :
                 (Default value = False)

            Returns
            -------

            """
            obs_tp1, r, done, _ = self.env.step(action)
            self.policy.add_obs(obs_tp1)

            if render:
                value_map = self.Q.copy()
                value_map = np.max(value_map, axis=2)
                render = self.env.render(value_map=value_map, density_map=self.get_density_map())

            _, action_tp1 = self.update_q_value(obs, action, r, obs_tp1, done)

            self.increment(obs, action, obs_tp1)
            # self.policy_map.add_node(obs_tp1, action)

            episode.append(obs, action, r, obs_tp1, done)

            return obs_tp1, action_tp1, done, render

    return SmartStart(env, *args, **kwargs)


if __name__ == "__main__":
    from smartstart.environments.gridworld import GridWorld, GridWorldVisualizer
    from smartstart.algorithms import SARSALambda

    directory = '/home/bartkeulen/repositories/smartstart/data/tmp'

    np.random.seed()

    env = GridWorld.generate(GridWorld.EASY)
    visualizer = GridWorldVisualizer(env)
    visualizer.add_visualizer(GridWorldVisualizer.LIVE_AGENT,
                              GridWorldVisualizer.CONSOLE,
                              GridWorldVisualizer.VALUE_FUNCTION,
                              GridWorldVisualizer.DENSITY)
    env.visualizer = visualizer
    # env.T_prob = 0.1

    # env.wall_reset = True

    agent = generate_smartstart_object(SARSALambda, env, eta=0.75, alpha=0.3, num_episodes=1000, max_steps=10000, exploitation_param=0.)

    summary = agent.train(render=True, render_episode=True)

    summary.save(directory=directory)
