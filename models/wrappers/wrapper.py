import torch.nn as nn


class Wrapper(nn.Module):
    def __init__(self, basenet, args):
        super(Wrapper, self).__init__()
        self.basenet = basenet

    def forward(self, x, meta):
        raise NotImplementedError()
