"""Module that contains random utility functions used in the smartstart package

"""
import os

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../',
                   'data/tmp')
if not os.path.exists(DIR):
    os.makedirs(DIR)


def get_data_directory(file):
    """Creates and returns a data directory at the file's location

    Parameters
    ----------
    file :
        python file

    Returns
    -------
    :obj:`str`
        filepath to the data directory

    """
    fp = os.path.dirname(os.path.abspath(file))
    fp = os.path.join(fp, 'data')
    if not os.path.exists(fp):
        os.makedirs(fp)
    return fp