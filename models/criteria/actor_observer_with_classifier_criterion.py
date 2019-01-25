import torch
import torch.nn as nn
from models.criteria.actor_observer_criterion import ActorObserverCriterion


def var_subset(var, inds):
    inds = torch.LongTensor(inds).to(var[0].device)
    out = [x.index_select(0, inds) for x in var]
    return out


class ActorObserverWithClassifierCriterion(ActorObserverCriterion):
    def __init__(self, args):
        super(ActorObserverWithClassifierCriterion, self).__init__(args)
        self.clsloss = nn.BCELoss(reduce=False)
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
            pred, f, targ = super(ActorObserverWithClassifierCriterion, self).forward(*vars1)
            final.append(f)
        else:
            pred = {'triplet_prediction': [], 'weights': []}
            targ = {'triplet_target': []}

        # Classification loss
        if len(inds2) > 0:
            cls2, target2 = var_subset([cls, -target.long()], inds2)
            clsloss = self.clsloss(nn.Sigmoid()(cls2), target2.float())
            clsloss = clsloss.mean(1)
            f = self.clsweight * clsloss.mean()
            final.append(f)
        else:
            cls2 = target2 = torch.Tensor([])

        print('losses:', ' '.join(['{}'.format(r.item()) for r in final]))
        pred['class_prediction'] = cls2.detach().cpu()
        targ['class_target'] = target2.detach().cpu()
        return pred, sum(final), targ
