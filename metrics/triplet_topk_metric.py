from metrics.triplet_accuracy_metric import TripletAccuracyMetric
import numpy as np


class TripletTopkMetric(TripletAccuracyMetric):
    def __init__(self):
        super(TripletTopkMetric, self).__init__()
        self.k = None

    def update(self, prediction, target):
        if isinstance(target, dict):
            target = target['triplet_target']
        if len(target) > 0:
            prec1 = self.triplet_topk(prediction['triplet_prediction'], target, prediction['weights'], topk=self.k)
            self.am.update(prec1.item())

    def __repr__(self):
        return 'triplet_top{}_metric {am.val:.3f} ({am.avg:.3f})'.format(self.k, am=self.am)

    def compute(self):
        return ('triplet_top{}_metric'.format(self.k), self.am.avg)

    def triplet_topk(self, output, target, weights, topk=5):
        weights = np.array(weights)
        n = weights.shape[0]
        topkn = int(np.ceil(.01 * topk * n))
        ind = np.argsort(weights)
        ind = ind[-topkn:].tolist()
        return self.triplet_accuracy([output[x] for x in ind], [target[x] for x in ind])
