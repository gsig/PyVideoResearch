from metrics.metric import Metric
from metrics.utils import AverageMeter, map
import numpy as np
import os
import re
import torch


def make_image_key(video_id, timestamp):
    """Returns a unique identifier for a video id & timestamp."""
    return "%s,%04d" % (video_id, int(timestamp))


def tensor2str(tensor):
    return [''.join([chr(y) for y in x]) for x in tensor]


class FRCNNMAPMetric(Metric):
    def __init__(self):
        self.am = AverageMeter()
        self.predictions = []
        self.targets = []
        this_dir = os.path.dirname(__file__)
        top60path = '../external/ActivityNet/Evaluation/ava/ava_action_list_v2.1_for_activitynet_2018.pbtxt.txt'
        top60path = os.path.join(this_dir, top60path)
        with open(top60path) as f:
            self.top60 = [int(x) for x in re.findall('[0-9]+', f.read())]

    def update(self, prediction, target):
        for t in target:
            gtlabels = t['labels']+1
            gt = torch.zeros(81)
            for g in gtlabels:
                g = int(g.item())
                gt[g] = 1
            self.targets.append(gt)

        for p in prediction:
            labels = p['labels'] + 1
            scores = p['scores']
            pred = torch.zeros(81)
            for l, s in zip(labels, scores):
                l = int(l.item())
                pred[l] = max(pred[l].item(), float(s))
            self.predictions.append(pred)

    #def __repr__(self):
    #    return '{}: {:.3f}'.format(*self.compute())

    def compute(self):
        mAP, _, ap = map(np.vstack(self.predictions), np.vstack(self.targets))
        print(ap)
        top60 = []
        for i in self.top60:
            m = ap[i]
            top60.append(m)
        return ('AVAmap', np.nanmean(top60))
