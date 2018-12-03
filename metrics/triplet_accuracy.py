from metrics.utils import AverageMeter
from metrics.metric import Metric
import numpy as np


class TripletAccuracy(Metric):
    def __init__(self):
        self.am = AverageMeter()

    def update(self, prediction, target):
        prec1 = self.triplet_accuracy(prediction['prediction'], target)
        self.am.update(prec1.item())

    def __repr__(self):
        return 'TripletAccuracy {am.val:.3f} ({am.avg:.3f})'.format(am=self.am)

    def compute(self):
        return ('TripletAccuracy', self.am.avg)

    def triplet_accuracy(self, output, target, weights=None):
        """
           if target>0 then first output should be smaller than right output
           optional weighted average
        """
        if type(output) is not list:
            output = [(x.data[0], y.data[0]) for x, y in zip(*output)]
        correct = [x < y if t > 0 else y < x for (x, y), t in zip(output, target)]
        if weights is None:
            return np.mean(correct)
        else:
            weights = weights.numpy()
            weights = weights / (1e-5 + np.sum(weights))
            return np.sum(np.array(correct).astype(float) * weights)
