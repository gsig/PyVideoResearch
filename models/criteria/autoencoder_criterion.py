# pylint: disable=W0221,E1101
import torch.nn as nn
from models.criteria.criterion import Criterion


class AutoencoderCriterion(Criterion):
    def __init__(self, args):
        super(AutoencoderCriterion, self).__init__(args)
        self.loss = nn.MSELoss()

    def forward(self, x_hat, code, x, target, meta, synchronous=False):
        loss = self.loss(x, x_hat)
        print('loss: {}'.format(loss.item()))

        return x_hat.detach().cpu(), loss, x.detach().cpu()
