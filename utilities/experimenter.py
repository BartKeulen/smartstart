from multiprocessing import Pool, cpu_count

from sklearn.model_selection import ParameterGrid


def run_experiment(param_grid, n_processes=-1):
    if type(param_grid) is not list:
        param_grid = [param_grid]

    for params in param_grid:
        if 'task' not in params:
            raise Exception('Please define a task function to execute.')

        if 'num_exp' not in params:
            params['run'] = [0]
        else:
            params['run'] = range(params['num_exp'])
            del params['num_exp']

        if type(params['task']) is not list:
            params['task'] = [params['task']]

    # Convert parameter grid to iterable list
    params = list(ParameterGrid(param_grid))
    for i in range(len(params)):
        params[i]['id'] = i

    if n_processes == -1:
        n_processes = cpu_count()
    if n_processes > 1:
        with Pool(n_processes) as p:
            p.map(process_task, params)
    else:
        for single_param in params:
            process_task(single_param)


def process_task(params):
    print("\033[1mProcess %d started\033[0m. Params: %s" % (params['id'], params))

    params['task'](params)

    print('\033[1mProcess %d finished\033[0m' % params['id'])