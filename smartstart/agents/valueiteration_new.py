import logging
import pdb
import random

import numpy as np

logger = logging.getLogger(__name__)


class ValueIteration:

    def __init__(self,
                 model,
                 v_init=0,
                 gamma=0.99,
                 min_error=1e-3,
                 max_itr=1000):
        self.model = model
        self.v_init = v_init
        self.V = [v_init for _ in range(self.model.num_states)]
        self.gamma = gamma
        self.min_error = min_error
        self.max_itr = max_itr

    def get_action(self, state):
        s_hash = self.model.hash_table.hash_state(state)
        V = []
        for a in range(self.model.num_actions):
            nV = np.array([self.V[ns] for ns in self.model.s_prime[s_hash][a]])
            P = np.array(self.model.P[s_hash][a])
            R = np.array(self.model.R[s_hash][a])
            V.append(np.dot(P, R + self.gamma * nV))

        return random.choice([i for i, v in enumerate(V) if v == max(V)])

    def optimize(self, state_list=None):
        d = self.model.num_states - len(self.V)
        if d > 0:
            self.V += [self.v_init]*d
        elif d < 0:
            raise Exception("More state-values than states in model, d = %d." % d)

        if state_list is None:
            state_list = range(self.model.num_states)

        delta = 0
        for itr in range(self.max_itr):
            delta = 0
            for s in state_list:
                if isinstance(s, np.ndarray):
                    s = self.model.hash_table(s)

                if s in self.model.terminal_states:
                    self.V[s] = 0.
                else:
                    V_old = self.V[s]

                    V = []
                    for a in range(self.model.num_actions):
                        nV = np.array([self.V[ns] for ns in self.model.s_prime[s][a]])
                        P = np.array(self.model.P[s][a])
                        R = np.array(self.model.R[s][a])
                        V.append(np.dot(P, R + self.gamma*nV))
                    self.V[s] = max(V)
                    delta = max(delta, abs(self.V[s] - V_old))
            if delta < self.min_error:
                logger.debug("VI Converged in %d iterations, delta: %.7f" % (itr, delta))
                return
        logger.debug("VI did not converge, delta: %.7f" % delta)


class TabularModel:

    def __init__(self, hash_table, num_actions, num_states=None):
        self.hash_table = hash_table
        self.num_states = 0
        self.num_actions = num_actions
        self.terminal_states = set()

        self.s_prime = []
        self.P = []
        self.R = []

        if num_states is not None:
            for i in range(num_states):
                self.s_prime.append([])
                self.P.append([])
                self.R.append([])
                for _ in range(self.num_actions):
                    self.s_prime[i].append([])
                    self.P[i].append([])
                    self.R[i].append([])
            self.num_states = num_states

    def set(self, s, a, ns, p, r, done=False, append=False):
        s_hash = self.hash_table.hash_state(s)
        ns_hash = self.hash_table.hash_state(ns)

        if done:
            self.add_terminal_state(ns)

        # If new states add to lists
        if s_hash >= self.num_states:
            self._add_state()
        if ns_hash >= self.num_states:
            self._add_state()

        if ns_hash not in self.s_prime[s_hash][a]:
            self.s_prime[s_hash][a].append(ns_hash)
            self.R[s_hash][a].append(r)
            self.P[s_hash][a].append(p)
        else:
            idx = self.s_prime[s_hash][a].index(ns_hash)
            if append:
                self.R[s_hash][a][idx] += r
                self.P[s_hash][a][idx] += p
            else:
                self.R[s_hash][a][idx] = r
                self.P[s_hash][a][idx] = p

    def _add_state(self):
        self.s_prime.append([])
        self.P.append([])
        self.R.append([])
        for _ in range(self.num_actions):
            self.s_prime[-1].append([])
            self.P[-1].append([])
            self.R[-1].append([])
        self.num_states += 1

    def get_transition(self, s, a=None, ns=None):
        s_hash = self.hash_table.hash_state(s)
        if s_hash >= self.num_states:
            return None

        if a is None:
            return self.P[s_hash]
        elif ns is None:
            return self.P[s_hash][a]
        else:
            ns_hash = self.hash_table.hash_state(ns)
            if ns_hash not in self.s_prime[s_hash][a]:
                return None
            else:
                idx = self.s_prime[s_hash][a].index(ns_hash)
            return self.P[s_hash][a][idx]

    def get_reward(self, s, a=None):
        s_hash = self.hash_table.hash_state(s)

        if s_hash >= self.num_states:
            return None

        if a is None:
            return self.R[s_hash]
        else:
            return self.R[s_hash][a]

    def clear(self, s, a=None):
        s_hash = self.hash_table.hash_state(s)
        if a is None:
            self.s_prime[s_hash] = []
            self.P[s_hash] = []
            self.R[s_hash] = []
            for _ in range(self.num_actions):
                self.s_prime[s_hash].append([])
                self.P[s_hash].append([])
                self.R[s_hash].append([])
        else:
            self.s_prime[s_hash][a] = []
            self.P[s_hash][a] = []
            self.R[s_hash][a] = []

    def add_terminal_state(self, state):
        s_hash = self.hash_table.hash_state(state)
        self.terminal_states.add(s_hash)

