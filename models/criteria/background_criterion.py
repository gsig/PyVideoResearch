# pylint: disable=W0221,E1101
import torch.nn as nn
from models.criteria.default_criterion import DefaultCriterion
from models.criteria.utils import winsmooth
import torch


class BackgroundCriterion(DefaultCriterion):
    def __init__(self, args):
        super(BackgroundCriterion, self).__init__(args)

    def forward(self, a, target, meta, synchronous=False):
        if len(target.shape) == 3:
            target = torch.cat([torch.zeros(target.shape[0], target.shape[1], 1).int().cuda(), target], dim=2)
            target[target.sum(2).sum(1) == 0, :, 0] = 1
        else:
            target = torch.cat([torch.zeros(target.shape[0], 1).int().cuda(), target], dim=1)
            target[target.sum(1) == 0, 0] = 1
        a, target, meta = self.process_tensors(a, target, meta, self.balance_loss)
        loss = self.loss(nn.Sigmoid()(a), target)
        print('losses class: {}'.format(loss.item()))

        if synchronous:
            a = winsmooth(a, kernelsize=self.win_smooth)
        return a, loss, target
