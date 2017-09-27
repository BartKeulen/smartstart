"""Module for schedulers

Schedulers can be used to gradually decrease a variable over time. The
scheduler will return a scale parameter that can be multiplied with the
original value.
"""


class Scheduler(object):
    """Base implementation of a Scheduler

    """

    def sample(self):
        """Returns the scale factor"""
        raise NotImplementedError


class LinearScheduler(Scheduler):
    """Returns a scale parameter that linearly decreases from start to end

    Every time the sample method is called the count is increased. When the
    count = start the scale will decrease from one to zero between
    count = start and count = end.

    Parameters
    ----------
    start : :obj:`int`
        start index of the scheduler
    end : :obj:`int`
        final index of the scheduler

    Attributes
    ----------
    start : :obj:`int`
        start index of the scheduler
    end : :obj:`int`
        final index of the scheduler
    count : :obj:`int`
        count
    scale : :obj:`float`
        current scale factor
    step : :obj:`float`
        step-size by which scale is decreased
    """

    def __init__(self, start, end):
        assert end > start
        self.start = start
        self.end = end
        self.count = 0
        self.scale = 1.
        self.step = 1. / (end - start)

    def sample(self):
        """Returns the scale factor"""
        if self.count > self.start and self.scale > 0.:
            self.scale -= self.step
        elif self.scale < 0.:
            self.scale = 0.

        self.count += 1

        return self.scale
