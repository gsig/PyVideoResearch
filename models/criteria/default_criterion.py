# pylint: disable=W0221,E1101
import torch
import torch.nn as nn
from torch.autograd import Variable
from models.layers.verbose_gradients import VerboseGradients
from models.layers.balance_labels import BalanceLabels
from models.layers.utils import gtmat
from models.criteria.utils import unroll_time, winsmooth
from criterion import Criterion


def _expand(arr, factor):
    return [x for x in arr for _ in range(factor)]


class DefaultCriterion(Criterion):
    def __init__(self, args):
        super(DefaultCriterion, self).__init__(args)
        self.orig_loss = args.originalloss_weight
        self.loss = nn.BCELoss()
        self.balanceloss = args.balanceloss
        self.BalanceLabels = BalanceLabels()
        self.winsmooth = args.window_smooth
        #self.videoloss = args.videoloss

    def process_tensors(self, a, target, id_time):
        if a.dim() == 3:
            # temporal mode
            a, target = unroll_time(a, target, self.training)
        elif target.dim() == 3:
            # temporal segment mode
            target = target.max(dim=1)[0]
        elif target.dim() == 1:
            print('converting Nx1 target to NxC')
            target = Variable(gtmat(a.shape, target.data.long()))
        target = target.float()
        a, = VerboseGradients.apply(a)
        if self.balanceloss and self.training:
            print('balancing loss')
            a = self.BalanceLabels(a, target)
        return a, target, id_time

    def forward(self, a, target, id_time, synchronous=False):
        a, target, id_time = self.process_tensors(a, target, id_time)
        loss = self.loss(torch.nn.Sigmoid()(a), target) * self.orig_loss
        #if self.videoloss:
        #    assert synchronous, 'videoloss only makes sense if there is only one video per batch'
        #    print('applying video loss')
        #    loss += self.loss(torch.max(torch.nn.Sigmoid()(a), dim=0)[0], torch.max(target, dim=0)[0]) * self.orig_loss
        print('losses class: {}'.format(loss.item()))

        if synchronous:
            a = winsmooth(a, kernelsize=self.winsmooth)
        return a, loss, target
