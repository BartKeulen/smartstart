import numpy as np


class Tabular:

    def __init__(self, env):
        self.hash_table = np.ones(env.grid_world.shape, int) * -1

        idx = 0
        self.terminal_states = []
        for i in range(env.grid_world.shape[0]):
            for j in range(env.grid_world.shape[1]):
                if env.grid_world[i, j] != 1:
                    self.hash_table[i, j] = idx
                    if env.grid_world[i, j] == 3 or env.grid_world[i, j] == 4:
                        self.terminal_states.append(idx)
                    idx += 1
        self.num_actions = env.num_actions
        self.num_states = idx

    def hash_state(self, state):
        idx = self.hash_table[tuple(state)]
        if idx < 0 or idx >= self.num_states:
            raise Exception("Hash index value of %d is out of range for state %s" % (idx, state))
        return idx

    def un_hash_state(self, idx):
        if idx < 0 or idx >= self.num_states:
            raise Exception("Hash index value of %d is out of range" % idx)

        state = np.where(self.hash_table == idx)
        return np.asarray([state[0][0], state[1][0]])

    def phi(self, state):
        idx = self.hash_state(state)
        phi = np.zeros(self.num_states, int)
        phi[idx] = 1
        return phi