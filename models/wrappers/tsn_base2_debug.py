"""
Temporal Segment Networks
from
Temporal Segment Networks: Towrads Good Practices for Deep Action Recognition
Liming Wang et al.
"""
from wrapper import Wrapper
import torch
import torch.nn as nn


class TSNBase2Debug(Wrapper):
    def __init__(self, basenet, args):
        self.consensus = lambda x: x.mean(1)
        super(TSNBase2Debug, self).__init__(basenet, args)
        nclasses, nhidden = args.nclass, args.nhidden
        self.nc = nclasses
        self.naa = nhidden
        self.mAAa = nn.Linear(1, self.nc * self.naa, bias=False)
        self.mAAb = nn.Linear(1, self.naa * self.nc, bias=False)

    def forward(self, x, meta):
        # input is b, t, ...
        s = x.shape
        #b, t = s[0], s[1]
        x = x.reshape(-1, *s[2:])
        out = self.basenet(x)
        #out = out.reshape(b, t, *out.shape[1:])
        const = torch.ones(out.shape[0], 1).cuda()

        aaa = self.mAAa(const).view(-1, self.nc, self.naa)
        aab = self.mAAb(const).view(-1, self.naa, self.nc)
        aa = torch.bmm(aaa, aab)
        #b = s[0]
        #out = out.reshape(b, 1, *out.shape[1:])
        #return self.consensus(out)
        return out, aa.view(-1, self.nc, self.nc)
