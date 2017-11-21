import gym

from smartstart.environments.environment import Environment


class GymEnv(Environment):

    def __init__(self, name):
        self._env = gym.make(name)
        self.obs_min = self._env.observation_space.low
        self.obs_max = self._env.observation_space.high
        self.num_states = self._env.observation_space.shape[0]
        self.num_actions = self._env.action_space.n
        super(GymEnv, self).__init__(name)

    def reset(self, start_state=None):
        return self._env.reset()

    def step(self, action):
        obs_tp1, reward, done, _ = self._env.step(action)

        return obs_tp1, reward, done

    def render(self, close=False):
        self._env.render(close=close)
