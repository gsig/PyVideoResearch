"""
   Normalize gradient of multiple inputs to have the same norm as the gradient of the first input
"""
from torch.autograd import Function


VERBOSE = True


def dprint(message, *args):
    if VERBOSE:
        print(message.format(*args))


class EqualizeGradNorm(Function):
    @staticmethod
    def forward(ctx, *inputs):
        output = [x.clone() for x in inputs]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grad_outputs):
        norms = [x.data.norm() + 1e-5 for x in grad_outputs]
        dprint('gradnorms before: {}', ' \t'.join([str(x.item()) for x in norms]))
        z = norms[0]
        grad_outputs = [x.clone() * (z / n) for x, n in zip(grad_outputs, norms)]
        dprint('gradnorms after : {}', ' \t'.join([str(x.norm().item()) for x in grad_outputs]))
        return tuple(grad_outputs)
