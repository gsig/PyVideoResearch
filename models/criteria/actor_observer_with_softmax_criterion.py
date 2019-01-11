import torch
import torch.nn as nn
from models.criteria.actor_observer_criterion import ActorObserverCriterion
import random


def var_subset(var, inds):
    inds = torch.LongTensor(inds).to(var[0].device)
    out = [x.index_select(0, inds) for x in var]
    return out


class ActorObserverWithSoftmaxCriterion(ActorObserverCriterion):
    def __init__(self, args):
        super(ActorObserverWithSoftmaxCriterion, self).__init__(args)
        self.clsloss = nn.CrossEntropyLoss(reduce=False)
        self.clsweight = args.classifier_weight

    def forward(self, dist_a, dist_b, x, y, z, cls, target, meta, synchronous=False):
        # target is a batch x n_class tensor
        # where the rows for ego triplets are one, padded by zeros
        # and class labels are negative
        ids = meta['id']
        inds1 = [i for i, t in enumerate(target) if t[0].item() > 0]
        inds2 = [i for i, t in enumerate(target) if t[0].item() <= 0]
        print('#triplets: {} \t #class: {}'.format(len(inds1), len(inds2)))
        final = []

        # ActorObserverLoss
        if len(inds1) > 0:
            vars1 = var_subset([dist_a, dist_b, x, y, z, target[:, 0]], inds1)
            vars1 += [{'id': [ids[i] for i in inds1]}]
            pred, f, targ = super(ActorObserverWithSoftmaxCriterion, self).forward(*vars1)
            final.append(f)
        else:
            pred = {'triplet_prediction': [], 'weights': []}
            targ = {'triplet_target': []}

        # Classification loss
        inds2 = [i for i in inds2 if target[i].sum() != 0]
        if len(inds2) > 0:
            cls2, target2 = var_subset([cls, -target.long()], inds2)
            b = target2.shape[0]
            oldsoftmax_target = torch.LongTensor(b).zero_()
            for i in range(b):
                if target2[i].sum() == 0:
                    oldsoftmax_target[i] = target2.shape[1]
                else:
                    oldsoftmax_target[i] = random.choice(target2[i].nonzero())
            target2 = oldsoftmax_target.to(target2.device)

            clsloss = self.clsloss(cls2, target2)
            f = self.clsweight * clsloss.sum()
            final.append(f)
        else:
            cls2 = target2 = torch.Tensor([])

        print('losses:', ' '.join(['{}'.format(r.item()) for r in final]))
        pred['class_prediction'] = nn.Softmax()(cls2.detach().cpu())
        targ['class_target'] = target2.detach().cpu()
        return pred, sum(final), targ
