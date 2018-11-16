import torch.nn as nn
import torch


class Criterion(nn.Module):
    def __init__(self, args):
        super(Criterion, self).__init__()

    def forward(self, a, target, meta, synchronous=False):
        raise NotImplementedError()
        loss = torch.Tensor(1)
        return a, loss, target
