import numpy as np


# Moving average
def moving_average(values, window=10):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')
