from metrics.utils import AverageMeter, map
from metrics.metric import Metric
import numpy as np


class MAPMetric(Metric):
    def __init__(self):
        self.am = AverageMeter()
        self.predictions = []
        self.targets = []

    def update(self, prediction, target):
        if target.dim() == 3:
            target = target.max(dim=1)[0]
        self.targets.append(target.max(dim=0)[0])
        prediction_video = prediction.max(dim=0)[0]
        self.predictions.append(prediction_video)

    def compute(self):
        mAP, _, ap = map(np.vstack(self.predictions), np.vstack(self.targets))
        print(ap)
        return ('mAP', mAP)
