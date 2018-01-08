"""Module that contains random utility functions used in the smartstart package

"""
import os


def get_data_directory(fp):
    """Creates and returns a data directory at the file's location

    Parameters
    ----------
    fp :
        python file

    Returns
    -------
    :obj:`str`
        filepath to the data directory

    """
    fp = os.path.abspath(fp)
    if not os.path.exists(fp):
        os.makedirs(fp)
    return fp