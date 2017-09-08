import pickle
import os


DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data/tmp')
if not os.path.exists(DIR):
    os.makedirs(DIR)


def deserialize(fp):
    data = open(fp, 'rb').read()
    return pickle.loads(data)


class Serializable(object):

    def __init__(self, *args):
        self.__args = args

    def serialize(self, filename, directory=DIR):
        fp = os.path.join(directory, filename + ".bin")
        f = open(fp, 'wb')
        f.write(pickle.dumps(self))
        return fp
