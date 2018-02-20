"""Module for making it easy to run experiments

This methods defined in this module make it easy to run multiple experiments
in parallel and using different sets of parameters.
"""
from multiprocessing import Pool, cpu_count
from collections import Mapping
from itertools import product

from tqdm import *


def run_experiment(param_grid, n_processes=None):
    """Method for running experiments

    This method is used for making it easy to run experiments in parallel and
    use different combinations of parameters. The param_grid dictionary
    describes what task or tasks must be executed and what parameters and/or
    parameter combinations must be used. Each experiment receives its own
    unique id.

    The param_grid consists of the following entries:
        * task : function that has to be executed
        * num_exp : number of time each experiment has to be executed (optional)
        * params : lists holding the params that have to be used

    Example of a param_grid:

        param_grid = {'task': task_func, 'num_exp': 5, 'alpha': [0.05, 0.1]}

    and the task func should then be in the following format:

        def task_func(params):
            # Define the code that has to be executed params defined can be
            # accessed like this
            alpha = params['alpha']

            # a run parameter, useful when num_exp > 1
            run = param['run']

            # and the unique id
            id = params['id']


    In total the task function will be executed 10 times. 5 times with each
    alpha parameter. See tutorial or examples for a full example.

    Parameters
    ----------
    param_grid : :obj:`dict`
        dictionary containing the parameters to be used during the experiments
    n_processes : :obj:`int`
        number of processes to run in parallel (Default value = -1,
        all the cpu_cores in the system are used)

    Raises
    ------
    Exception
        Please define a task function in the param_grid to execute.
    """
    if isinstance(param_grid, Mapping):
        # wrap dictionary in a singleton list to support either dict
        # or list of dicts
        param_grid = [param_grid]

    for params in param_grid:
        if 'task' not in params:
            raise Exception('Please define a task function in the param_grid '
                            'to execute.')

        if 'num_exp' not in params:
            params['run'] = [0]
        else:
            params['run'] = range(params['num_exp'])
            del params['num_exp']

    # Convert parameter grid to iterable list
    params = list(parameter_grid(param_grid))
    for i in range(len(params)):
        params[i]['id'] = i

    print("\n\033[1mNumber of processes: %d\033[0m" % len(params))

    if n_processes is None or n_processes <= 0:
        n_processes = cpu_count()
    if n_processes > 1:
        with Pool(n_processes) as p:
            r = list(tqdm(p.imap_unordered(process_task, params), total=len(params)))
    else:
        for single_param in params:
            process_task(single_param)


def process_task(params):
    """Helper method for executing the task defined in params

    Parameters
    ----------
    params : :obj:`dict`
        dictionary with the parameters to be used in the experiment
    """
    params['task'](params)


def parameter_grid(param_grid):
    for p in param_grid:
        # Always sort the keys of a dictionary, for reproducibility
        items = sorted(p.items())
        if not items:
            yield {}
        else:
            keys, values = zip(*items)
            values = list(values)
            for i, v in enumerate(values):
                if not isinstance(v, list):
                    if hasattr(v, '__iter__'):
                        values[i] = list(v)
                    else:
                        values[i] = [v]
            for v in product(*values):
                params = dict(zip(keys, v))
                yield params