from misc_utils.utils import AverageMeter
from metrics.utils import map
from metrics.metric import Metric
import numpy as np


class VideomAPMetric(Metric):
    def __init__(self):
        self.am = AverageMeter()
        self.predictions = []
        self.targets = []

    def update(self, prediction, target):
        if target.dim() == 3:
            target = target.max(dim=1)[0]
        self.targets.append(target)
        self.predictions.append(prediction)

    def compute(self):
        mAP, _, ap = map.map(np.vstack(self.predictions), np.vstack(self.targets))
        print(ap)
        return ('framemAP', mAP)
