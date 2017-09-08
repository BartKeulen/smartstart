import os


def get_data_directory(file):
    fp = os.path.dirname(os.path.abspath(file))
    fp = os.path.join(fp, 'data')
    if not os.path.exists(fp):
        os.makedirs(fp)
    return fp