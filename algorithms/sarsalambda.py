import numpy as np

from algorithms.sarsa import SARSA
from smartstart.smartstart_file import SmartStart


class SARSALambda(SARSA):

    def __init__(self, env, lamb=0.9, threshold_traces=1e-3, *args, **kwargs):
        super(SARSALambda, self).__init__(env, *args, **kwargs)
        self.lamb = lamb
        self.threshold_traces = threshold_traces
        self.traces = np.zeros((self.env.w, self.env.h, self.env.num_actions))

    def update_q_value(self, obs, action, td_error):
        idx = tuple(obs) + (action,)
        self.traces[idx] = 1
        active_traces = np.asarray(np.where(self.traces > self.threshold_traces))
        for i in range(active_traces.shape[1]):
            idx = tuple(active_traces[:, i])
            self.Q[idx] += self.alpha * td_error * self.traces[idx]
            self.traces[idx] *= self.gamma * self.lamb