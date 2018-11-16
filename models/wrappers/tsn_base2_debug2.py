"""
Temporal Segment Networks
from
Temporal Segment Networks: Towrads Good Practices for Deep Action Recognition
Liming Wang et al.
"""
from wrapper import Wrapper


class TSNBase2Debug(Wrapper):
    def __init__(self, basenet, args):
        self.consensus = lambda x: x.mean(1)
        super(TSNBase2Debug, self).__init__(basenet, args)

    def forward(self, x, meta):
        # input is b, t, ...
        s = x.shape
        out = self.basenet(x)
        out = out.unsqueeze(1)
        #b = s[0]
        #out = out.reshape(b, 1, *out.shape[1:])
        #return self.consensus(out)
        return out
