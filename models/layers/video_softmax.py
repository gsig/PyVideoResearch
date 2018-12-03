import torch
from torch.autograd import Function
from torch import nn
import math


class MulConstant(Function):
    @staticmethod
    def forward(ctx, tensor, constant):
        ctx.constant = constant
        return tensor * constant

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.constant, None


def lsetwo(a, b):
    log, exp = math.log, math.exp
    return log(1 + exp(-abs(a - b))) + max(a, b)


class VideoSoftmax(nn.Module):
    def __init__(self, storage, decay):
        super(VideoSoftmax, self).__init__()
        self.storage = storage
        self.decay = decay

    def get_constants(self, ids):
        out = [self.storage[x] for x in ids]
        # out = [math.exp(-x) for x in ids]
        return torch.autograd.Variable(torch.Tensor(out).cuda())

    def update_constants(self, input, ids):
        for x, vid in zip(input, ids):
            if vid not in self.storage:
                self.storage[vid] = x.data[0]
            else:
                a = math.log(self.decay) + self.storage[vid]
                b = math.log(1 - self.decay) + x.data[0]
                self.storage[vid] = lsetwo(a, b)

    def forward(self, input, ids):
        self.update_constants(input, ids)
        constants = self.get_constants(ids)
        x = (input - constants).exp()
        return x
