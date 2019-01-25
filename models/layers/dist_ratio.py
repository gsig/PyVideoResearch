"""
   Probabilistic Criterion for Triplet Siamese Model for learning embedding.
   Ref: https://arxiv.org/pdf/1610.00243.pdf

   loss = -log( exp(-X) / ( exp(-X) + exp(-Y) ) )
   where
   X : Distance between similar samples
   Y : Distance between dissimilar samples

   The loss could be break down to following log expansion

   loss = -log( exp(-X) ) - (-log( exp(-X) + exp(-Y) ))
        = -log( exp(-X) ) + log( exp(-X) + exp(-Y) )
        = -(-X) + log( exp(-X) + exp(-Y) )
        = X + log( exp(-X) + exp(-Y) )

   Gradients:
      dLoss/dX = 1 + 1 / (exp(-X) + exp(-Y)) * -1 * exp(-X)
               = 1 - exp(-X) / (exp(-X) + exp(-Y))

      dLoss/dY = 0 + 1 / (exp(-X) + exp(-Y)) * -1 * exp(-Y)
               = -exp(-Y) / (exp(-X) + exp(-Y))
"""
import torch
from torch.autograd import Function, Variable


class DistRatio(Function):
    @staticmethod
    def forward(ctx, input1, input2, y, margin):
        """
        X : Distance between similar samples
        Y : Distance between dissimilar samples
        loss = -log( exp(-X) ) - (-log( exp(-X) + exp(-Y) ))
             = -log( exp(-X) ) + log( exp(-X) + exp(-Y) )
             = -(-X) + log( exp(-X) + exp(-Y) )
             = X + log( exp(-X) + exp(-Y) )
        """
        exp, log = torch.exp, torch.log
        ctx.margin = margin
        _output = input1.clone()
        #_output[y == -1] = input2[y == -1].clone()  # supporting switched samples
        _output.add_(log(exp(-input1) + exp(-input2)))
        output = _output
        ctx.save_for_backward(input1, input2, y)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        X : Distance between similar samples
        Y : Distance between dissimilar samples

        Gradients:
           dLoss/dX = 1 + 1 / (exp(-X) + exp(-Y)) * -1 * exp(-X)
                    = 1 - exp(-X) / (exp(-X) + exp(-Y))

           dLoss/dY = 0 + 1 / (exp(-X) + exp(-Y)) * -1 * exp(-Y)
                    = -exp(-Y) / (exp(-X) + exp(-Y))
        """
        input1, input2, y = ctx.saved_variables
        grad_input1 = Variable(input1.data.new(input1.size()).zero_())
        grad_input2 = Variable(input1.data.new(input1.size()).zero_())

        grad_input1.add_(-1, torch.exp(-input1.clone()))
        grad_input2.add_(-1, torch.exp(-input2.clone()))

        dist = input1.clone().mul(-1).exp() + input2.clone().mul(-1).exp()
        grad_input1.div_(dist)
        grad_input2.div_(dist)

        grad_input1.add_(1)
        #grad_input1[y == 1].add_(1)  # supporting switched samples
        #grad_input2[y == -1].add_(1)  # supporting switched samples

        return grad_input1 * grad_output, grad_input2 * grad_output, None, None

