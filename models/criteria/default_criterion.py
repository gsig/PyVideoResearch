# pylint: disable=W0221,E1101
import torch.nn as nn
from models.layers.verbose_gradients import VerboseGradients
from models.layers.balance_labels import BalanceLabels
from models.layers.utils import gtmat
from models.criteria.utils import unroll_time, winsmooth
from models.criteria.criterion import Criterion


class DefaultCriterion(Criterion):
    def __init__(self, args):
        super(DefaultCriterion, self).__init__(args)
        self.loss = nn.BCELoss()
        self.balance_loss = args.balanceloss
        self.balance_labels = BalanceLabels()
        self.win_smooth = args.window_smooth

    def process_tensors(self, a, target, meta, balance=False):
        if a.dim() == 3:
            # temporal mode
            print('converting NxTxC a+target to N2xC')
            a, target = unroll_time(a, target, self.training)
        elif target.dim() == 3:
            # temporal segment mode
            print('converting NxTxC target to NxC')
            target = target.max(dim=1)[0]
        elif target.dim() == 1:
            print('converting Nx1 target to NxC')
            target = gtmat(a.shape, target.detach().long())
        target = target.float()
        a, = VerboseGradients.apply(a)
        if balance and self.training:
            print('balancing loss')
            a = self.balance_labels(a, target)
        return a, target, meta

    def forward(self, a, target, meta, synchronous=False):
        a, target, meta = self.process_tensors(a, target, meta, self.balance_loss)
        loss = self.loss(nn.Sigmoid()(a), target)
        print('losses class: {}'.format(loss.item()))

        if synchronous:
            a = winsmooth(a, kernelsize=self.win_smooth)
        return a.detach().cpu(), loss, target.detach().cpu()
