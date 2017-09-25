

class Scheduler(object):

    def sample(self):
        raise NotImplementedError


class LinearScheduler(Scheduler):

    def __init__(self, start, end):
        assert end > start
        self.start = start
        self.end = end
        self.count = 0
        self.step = 1. / (end - start)
        self.scale = 1.

    def sample(self):
        if self.count > self.start and self.scale > 0.:
            self.scale -= self.step
        elif self.scale < 0.:
            self.scale = 0.

        self.count += 1

        return self.scale
