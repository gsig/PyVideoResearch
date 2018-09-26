from torch.autograd import Function, Variable
import torch.nn as nn
import torch


def populate(dict, ind, val=0):
    if ind not in dict:
        dict[ind] = val


class ScaleGrad(Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        ctx.save_for_backward(inputs, weights)
        return inputs.clone()

    @staticmethod
    def backward(ctx, grad_output):
        _, weights = ctx.saved_variables
        return grad_output * weights, None


class BalanceLabels(nn.Module):
    def __init__(self):
        super(BalanceLabels, self).__init__()
        self.zerocounts = {}
        self.counts = {}
        self.total = 0

    def update_counts(self, target):
        n = target.shape[0]
        tt = target.sum(0)
        for j, t in enumerate(tt):
            populate(self.counts, j)
            populate(self.zerocounts, j)
            self.counts[j] += t.item()
            self.zerocounts[j] += n - t.item()
        self.total += n

    def get_weights(self, target):
        weights = torch.zeros(*target.shape)
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                if target[i, j].item() == 0:
                    weights[i, j] = self.zerocounts[j]
                else:
                    weights[i, j] = self.counts[j]
        avg = self.total / 2
        return Variable(avg / weights)

    def forward(self, inputs, target):
        self.update_counts(target)
        weights = self.get_weights(target)
        return ScaleGrad.apply(inputs, weights.cuda())
