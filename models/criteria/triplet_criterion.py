from models.criteria.criterion import Criterion
from models.layers.dist_ratio import DistRatio


class ActorObserverCriterion(Criterion):
    def __init__(self, args):
        super(ActorObserverCriterion, self).__init__(args)
        self.loss = DistRatio()
        self.margin = args.margin

    def forward(self, dist_a, dist_b, x, y, z, target, meta, synchronous=False):
        ids = meta['id']
        w = x * y * z
        loss = self.loss.apply(dist_a, dist_b, target, self.margin)
        self.update_constants(loss, w, ids)

        print('loss before', loss.sum().item())

        pred = {'triplet_prediction': [(a, b) for a, b in zip(dist_a.detach().cpu(), dist_b.detach().cpu())],
                'weights': w.detach().cpu()}
        targ = {'triplet_target': target.cpu()}
        return pred, loss, targ
