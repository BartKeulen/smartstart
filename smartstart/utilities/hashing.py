import numpy as np


class HashTable:

    def __init__(self):
        self.hash_table = dict()
        self.un_hash_table = []
        self.idx = 0

    def reset(self):
        self.hash_table.clear()
        self.idx = 0

    def hash_state(self, state):
        key = tuple(state)
        if key in self.hash_table:
            return self.hash_table[key]
        else:
            self.hash_table[key] = self.idx
            self.idx += 1
            self.un_hash_table.append(key)
            return self.idx - 1

    def un_hash_state(self, s_hash):
        if s_hash < 0 or s_hash >= self.idx:
            raise Exception("A hash value of %d is out of range." % s_hash)

        return np.asarray(self.un_hash_table[s_hash])

    def phi(self, state, num_states=None):
        if num_states is None:
            num_states = self.idx
        idx = self.hash_state(state)
        phi = np.zeros(num_states, int)
        phi[idx] = 1
        return phi