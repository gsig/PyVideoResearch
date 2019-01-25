"""
ActorObserver Base model
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from models.wrappers.default_wrapper import DefaultWrapper
from models.utils import remove_last_layer
import random


class ActorObserverWrapper(DefaultWrapper):
    def __init__(self, basenet, opts, *args, **kwargs):
        super(ActorObserverWrapper, self).__init__(basenet, opts, *args, **kwargs)
        remove_last_layer(self.basenet)
        dim = basenet.in_features
        self.firstpos_fc = nn.Sequential(nn.Linear(dim, 1), nn.Tanh())
        self.firstpos_scale = nn.Parameter(torch.Tensor([math.log(.5)]))
        self.third_fc = nn.Sequential(nn.Linear(dim, 1), nn.Tanh())
        self.third_scale = nn.Parameter(torch.Tensor([math.log(.5)]))
        self.distance = opts.distance
        if opts.share_selector:
            self.firstneg_fc = self.firstpos_fc
            self.firstneg_scale = self.firstpos_scale
        else:
            self.firstneg_fc = nn.Sequential(nn.Linear(dim, 1), nn.Tanh())
            self.firstneg_scale = nn.Parameter(torch.Tensor([math.log(.5)]))

    def base(self, x, y, z):
        #base_y = self.basenet(y)
        #if random.random() > .5:  # TODO Debug, make sure order doesn't matter
        #    base_x = self.basenet(x)
        #    base_z = self.basenet(z)
        #else:
        #    base_z = self.basenet(z)
        #    base_x = self.basenet(x)
        base_x = self.basenet(x)
        base_y = self.basenet(y)
        base_z = self.basenet(z)

        if self.distance == 'cosine':
            dist_a = .5 - .5 * F.cosine_similarity(base_x, base_y, 1, 1e-6).view(-1)
            dist_b = .5 - .5 * F.cosine_similarity(base_y, base_z, 1, 1e-6).view(-1)
        elif self.distance == 'l2':
            dist_a = F.pairwise_distance(base_x, base_y, 2).view(-1)
            dist_b = F.pairwise_distance(base_y, base_z, 2).view(-1)
        else:
            assert False, "Wrong args.distance"

        print('fc7 norms:', base_x.norm().item(), base_y.norm().item(), base_z.norm().item())
        print('pairwise dist means:', dist_a.mean().item(), dist_b.mean().item())
        return base_x, base_y, base_z, dist_a, dist_b

    def verbose(self):
        print('scales:',
              math.exp(self.firstpos_scale.item()),
              math.exp(self.third_scale.item()),
              math.exp(self.firstneg_scale.item()))

    def forward(self, inputs, meta):
        if self.freeze_batchnorm:
            for module in self.basenet.modules():
                if isinstance(module, torch.nn.modules.BatchNorm1d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm2d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm3d):
                    module.eval()
        x, y, z = inputs
        """ assuming:
            x: first person positive
            y: third person
            z: first person negative
        """
        base_x, base_y, base_z, dist_a, dist_b = self.base(x, y, z)
        w_x = self.firstpos_fc(base_x).view(-1) * torch.exp(self.firstpos_scale)
        w_y = self.third_fc(base_y).view(-1) * torch.exp(self.third_scale)
        w_z = self.firstneg_fc(base_z).view(-1) * torch.exp(self.firstneg_scale)
        self.verbose()
        return dist_a, dist_b, w_x, w_y, w_z
