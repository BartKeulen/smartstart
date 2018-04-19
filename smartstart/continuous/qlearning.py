from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import super
from future import standard_library
standard_library.install_aliases()
from past.utils import old_div
from rlpy.Agents.Agent import Agent, DescentAlgorithm
from rlpy.Tools import addNewElementForAllActions, count_nonzero
import numpy as np


class QLearning(DescentAlgorithm, Agent):

    def __init__(self, policy, representation, discount_factor, **kwargs):
        super(QLearning, self).__init__(policy=policy, representation=representation,
                                        discount_factor=discount_factor, **kwargs)

    def _future_action(self, ns, terminal, np_actions, ns_phi, na):
        return self.representation.bestAction(ns, terminal, np_actions, ns_phi)

    def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):
        # The previous state could never be terminal
        # (otherwise the episode would have already terminated)
        prevStateTerminal = False

        # MUST call this at start of learn()
        self.representation.pre_discover(s, prevStateTerminal, a, ns, terminal)

        # Compute feature function values and next action to be taken

        discount_factor = self.discount_factor  # 'gamma' in literature
        weight_vec = self.representation.weight_vec  # Value function, expressed as feature weights
        phi_s = self.representation.phi(s, prevStateTerminal)  # active feats in state
        phi = self.representation.phi_sa(s, prevStateTerminal, a, phi_s)  # active features or an (s,a) pair
        phi_prim_s = self.representation.phi(ns, terminal)
        na = self._future_action(ns, terminal, np_actions, phi_prim_s, na)
        phi_prime = self.representation.phi_sa(ns, terminal, na, phi_prim_s)
        nnz = count_nonzero(phi_s)  # Number of non-zero elements

        # Compute td-error
        td_error = r + np.dot(discount_factor * phi_prime - phi, weight_vec)

        # Update value function (or if TD-learning diverges, take no action)
        if nnz > 0:
            weight_vec_old = weight_vec.copy()
            weight_vec += self.learn_rate * self.representation.featureLearningRate() * td_error * phi
            if not np.all(np.isfinite(weight_vec)):
                weight_vec = weight_vec_old
                print("WARNING: TD-Learning diverged, theta reached infinity!")

        # MUST call this at end of learn() - add new features to representation as required.
        expanded = self.representation.post_discover(s, False, a, td_error, phi_s)

        # MUST call this at end of learn() - handle episode termination cleanup as required.
        if terminal:
            self.episodeTerminated()