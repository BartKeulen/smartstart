import logging
import pdb

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
        self.representation = model.representation
        self.V = [v_init for _ in range(self.representation.num_states)]
        self.gamma = gamma
        self.min_error = min_error
        self.max_itr = max_itr

    def optimize(self):
        if not self.model.model_change:
            logger.debug("Model has not changed.")
            return

        self.model.model_change = False
        delta = 0
        for itr in range(self.max_itr):
            delta = 0
            for s in range(self.representation.num_states):
                if s not in self.representation.terminal_states:
                    V_old = self.V[s]

                    V = []
                    for a in range(self.representation.num_actions):
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

    def __init__(self, representation, tol=1e-3):
        self.representation = representation
        self.model_change = False
        self.tol = tol

        self.s_prime = []
        self.P = []
        self.R = []
        self.C = []
        self.R_sum = []
        for i in range(self.representation.num_states):
            self.s_prime.append([])
            self.P.append([])
            self.R.append([])
            self.C.append([])
            self.R_sum.append([])
            for _ in range(self.representation.num_actions):
                self.s_prime[i].append([])
                self.P[i].append([])
                self.R[i].append([])
                self.C[i].append([])
                self.R_sum[i].append([])

    def add_transition(self, s, a, ns, r):
        s_hash = self.representation.hash_state(s)
        ns_hash = self.representation.hash_state(ns)
        if ns_hash not in self.s_prime[s_hash][a]:
            self.s_prime[s_hash][a].append(ns_hash)
            self.C[s_hash][a].append(1)
            self.R_sum[s_hash][a].append(r)
            self.R[s_hash][a].append(r)
            self.P[s_hash][a].append(1 / sum(self.C[s_hash][a]))
            self.model_change = True
        else:
            idx = self.s_prime[s_hash][a].index(ns_hash)
            R_old = self.R_sum[s_hash][a][idx]
            P_old = self.P[s_hash][a][idx]
            self.C[s_hash][a][idx] += 1
            self.R_sum[s_hash][a][idx] += r
            self.R[s_hash][a][idx] = self.R_sum[s_hash][a][idx] / self.C[s_hash][a][idx]
            self.P[s_hash][a][idx] = self.C[s_hash][a][idx] / sum(self.C[s_hash][a])
            if abs(self.P[s_hash][a][idx] - P_old) > self.tol or abs(self.R[s_hash][a][idx] - R_old) > self.tol:
                self.model_change = True
            else:
                self.model_change = False

    def set(self, s, a, ns, p, r):
        self.model_change = True
        s_hash = self.representation.hash_state(s)
        ns_hash = self.representation.hash_state(ns)
        if ns_hash not in self.s_prime[s_hash][a]:
            self.s_prime[s_hash][a].append(ns_hash)
            self.R[s_hash][a].append(r)
            self.P[s_hash][a].append(p)
        else:
            idx = self.s_prime[s_hash][a].index(ns_hash)
            self.R[s_hash][a][idx] += r
            self.P[s_hash][a][idx] += p
