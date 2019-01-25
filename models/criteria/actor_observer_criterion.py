import torch
from models.criteria.criterion import Criterion
from models.layers.equalize_grad_norm import EqualizeGradNorm
from models.layers.video_softmax import VideoSoftmax
from models.layers.dist_ratio import DistRatio


class ActorObserverCriterion(Criterion):
    def __init__(self, args):
        super(ActorObserverCriterion, self).__init__(args)
        self.loss = DistRatio()
        self.xstorage = {}
        self.ystorage = {}
        self.zstorage = {}
        self.storage = {}
        self.decay = args.finaldecay
        self.xmax = VideoSoftmax(self.xstorage, args.decay)
        self.ymax = VideoSoftmax(self.ystorage, args.decay)
        if args.share_selector:
            self.zmax = self.xmax
        else:
            self.zmax = VideoSoftmax(self.zstorage, args.decay)
        self.margin = args.margin
        self.normalize_per_video = args.normalize_per_video

    def get_constants(self, ids):
        if not self.normalize_per_video:
            ids = ['all' for x in ids]
        out = [self.storage[x][0] for x in ids]
        return torch.Tensor(out)

    def update_constants(self, input, weights, ids):
        if not self.normalize_per_video:
            ids = ['all' for x in ids]
        for x, w, vid in zip(input, weights, ids):
            x, w = x.item(), w.item()
            if vid not in self.storage:
                self.storage[vid] = [x, w]
            else:
                # here J is stored as E[wJ]
                old_x, old_w = self.storage[vid]
                val = (1 - self.decay) * w * x + self.decay * old_w * old_x
                new_weight = (1 - self.decay) * w + self.decay * old_w
                val = val / new_weight
                self.storage[vid] = [val, new_weight]
                if new_weight < 0.0001:
                    print('MILC new_weight is effectively 0')

    def forward(self, dist_a, dist_b, x, y, z, target, meta, synchronous=False):
        # Normalize and combine weights
        ids = meta['id']
        y = self.ymax(y, ids)
        # since xmax and zmax might be the same we want to first update constants
        # and then apply the layer such that order does not matter
        self.xmax(x, ids)
        self.zmax(z, ids)
        x = self.xmax(x, ids, update_constants=False)
        z = self.zmax(z, ids, update_constants=False)
        dist_a, dist_b, x, y, z = EqualizeGradNorm.apply(dist_a, dist_b, x, y, z)
        w = x * y * z

        # update L
        loss = self.loss.apply(dist_a, dist_b, target, self.margin)
        #loss = torch.max(torch.zeros(loss.shape).to(loss.device), loss)  # to avoid underflow TODO
        self.update_constants(loss, w, ids)
        k = self.get_constants(ids).to(dist_a.device)
        n = (w.sum() + 0.00001) / w.shape[0]
        final = ((loss - k) * (w / n)).sum()
        # final += (loss * w / n).sum().detach()  # for loss curves

        print('loss before', loss.sum().item())
        print('loss after', (loss * w / n).sum().item())
        print('weight median: {}, var: {}'.format(w.median().item(), w.var().item()))

        pred = {'triplet_prediction': [(a, b) for a, b in zip(dist_a.detach().cpu(), dist_b.detach().cpu())],
                'weights': w.detach().cpu()}
        targ = {'triplet_target': target.cpu()}
        return pred, final, targ
