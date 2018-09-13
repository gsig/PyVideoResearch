"""
   usage:
   y, = BlockGradient(x)
   or: 
   y1,y2 = BlockGradient(x1,x2)
"""
from torch.autograd import Function, Variable


class BlockGradient(Function):
    @staticmethod
    def forward(ctx, *inputs):
        output = [x.clone() for x in inputs]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_outputs = [x.clone().zero_() for x in grad_outputs]
        return tuple(grad_outputs)
