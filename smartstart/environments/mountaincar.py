import numpy as np

from smartstart.environments.environment import Environment


class MountainCar(Environment):

    def __init__(self):
        super(MountainCar, self).__init__('MountainCar')
        self.obs_min = np.array([-1.2, -0.07])
        self.obs_max = np.array([0.5, 0.07])
        self.num_states = 2
        self.actions = [0, 1, 2]
        self._control_input = [0, -1, 1]
        self.num_actions = len(self.actions)
        self.state = None

    def reset(self, start_state=None):
        self.state = np.random.rand(2) * (self.obs_max - self.obs_min) + self.obs_min
        return self.state

    def step(self, action):
        if action not in self.actions:
            raise NotImplementedError("Action %d is not available. Please choose from the available actions: %s"
                                      % (action, self.actions))

        xdot = self.state[1] + 0.001 * self._control_input[action] - 0.0025 * np.cos(3 * self.state[0])
        xdot = min(max(xdot, self.obs_min[1]), self.obs_max[1])
        xnew = self.state[0] + xdot
        done = False
        if xnew <= self.obs_min[0]:
            xdot = 0.
        elif xnew >= self.obs_max[0]:
            done = True
        xnew = min(max(xnew, self.obs_min[0]), self.obs_max[0])

        self.state = np.asarray([xnew, xdot])
        return self.state, -1, done


if __name__ == "__main__":
    env = MountainCar()

    import pdb
    pdb.set_trace()

    obs = env.reset()
    print(obs)
    while True:
        action = int(input())

        obs, reward, done = env.step(action)

        print(obs, reward)
        if done:
            break

    print("DONE!")