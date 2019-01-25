"""
   Use ActorObserver model with classifier output
"""
import torch
import torch.nn as nn
from models.wrappers.actor_observer_wrapper import ActorObserverWrapper


class ActorObserverWithClassifierWrapper(ActorObserverWrapper):
    def __init__(self, basenet, opts, *args, **kwargs):
        super(ActorObserverWithClassifierWrapper, self).__init__(basenet, opts, *args, **kwargs)
        dim = basenet.in_features
        self.classifier = nn.Linear(dim, opts.nclass)

    def forward(self, inputs, meta):
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
        cls = self.classifier(base_y)
        self.verbose()
        return dist_a, dist_b, w_x, w_y, w_z, cls
