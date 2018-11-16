from mics_utils.utils import AverageMeter
from metrics.metric import Metric

class MockMetric(Metric):
    def __init__(self):
        self.am = AverageMeter()

    def update(self, prediction, target):
        self.am.update(prediction.mean())

    def __repr__(self):
        return '{} value: {}'.format(self.__class__.__name__, self.am.avg)

    def compute(self):
        return ('MockMetric', self.am.avg)
