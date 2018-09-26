# pylint: disable=W0221,E1101
import torch
import torch.nn as nn
from models.criteria.utils import winsmooth
import random
from default_criterion import DefaultCriterion


class SoftmaxCriterion(DefaultCriterion):
    def __init__(self, args):
        super(SoftmaxCriterion, self).__init__(args)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, a, target, id_time, synchronous=False):
        a, target, id_time = self.process_tensors(a, target, id_time)

        b = target.shape[0]
        softmax_target = torch.LongTensor(b).zero_()
        for i in range(b):
            if target[i].sum() == 0:
                softmax_target[i] = b
            else:
                softmax_target[i] = random.choice(target[i].nonzero())
        loss = self.loss(a, softmax_target.cuda()) * self.orig_loss
        print('losses class: {}'.format(loss.item()))

        if synchronous:
            a = winsmooth(a, kernelsize=self.winsmooth)
        return a, loss, target
