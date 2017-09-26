"""Module for numerical operations

"""
import numpy as np


# Moving average
def moving_average(values, window=10):
    """Moving average filter

    Parameters
    ----------
    values : :obj:`np.ndarray`
        array the moving average filter has to applied on
    window : :obj:`int`
        window size for the moving average filter (Default value = 10)

    Returns
    -------
    :obj:`np.ndarray`
        values after applying moving average filter (length = orignal_length
        - window + 1)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')
