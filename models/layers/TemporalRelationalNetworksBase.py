"""
Temporal Relational Networks 
from
Temporal Relational Reasoning in Videos
by Bolei Zhou, Alex Andonian, Aude Oliva, Antonio Torralba
"""
import torch.nn as nn
import torch
from torch.autograd import Variable


class TemporalRelationalNetworksBase(nn.Module):
    def __init__(self, basenet, dim, args):
        super(AsyncTFBase, self).__init__()
        self.basenet = basenet
        self.nc = nclasses
        self.naa = nhidden
        self.mAAa = nn.Linear(1, self.nc * self.naa, bias=False)
        self.mAAb = nn.Linear(1, self.naa * self.nc, bias=False)

    def forward(self, x):
        # input is b, c, t, h, w
        n = 2
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(-1, n, c, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        out = self.basenet(x)
        # TODO

        return out
