# pylint: disable=W0221,E1101
import torch
import torch.nn as nn
from models.criteria.utils import winsmooth
import random
from models.criteria.default_criterion import DefaultCriterion


class SoftmaxCriterion(DefaultCriterion):
    def __init__(self, args):
        super(SoftmaxCriterion, self).__init__(args)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, a, target, meta, synchronous=False):
        a, target, meta = self.process_tensors(a, target, meta)

        b = target.shape[0]
        oldsoftmax_target = torch.LongTensor(b).zero_()
        for i in range(b):
            if target[i].sum() == 0:
                oldsoftmax_target[i] = target.shape[1]
            else:
                oldsoftmax_target[i] = random.choice(target[i].nonzero())
        #target = target.max(1)[0]
        softmax_target = target.nonzero()[:, 1].long()

        loss = self.loss(a, softmax_target.cuda())
        print('losses class: {}'.format(loss.item()))

        if synchronous:
            a = winsmooth(a, kernelsize=self.win_smooth)
        return a.detach().cpu(), loss, target.detach().cpu()
