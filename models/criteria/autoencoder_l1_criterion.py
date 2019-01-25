# pylint: disable=W0221,E1101
import torch.nn as nn
from models.criteria.criterion import Criterion


class AutoencoderL1Criterion(Criterion):
    def __init__(self, args):
        super(AutoencoderL1Criterion, self).__init__(args)
        self.loss = nn.L1Loss()

    def forward(self, x_hat, code, x, target, meta, synchronous=False):
        loss = self.loss(x, x_hat)
        print('loss: {}'.format(loss.item()))

        return x_hat.detach().cpu(), loss, x.detach().cpu()
