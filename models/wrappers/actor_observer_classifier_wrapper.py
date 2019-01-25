"""
   Use ActorObserver model as first person classifier
"""
import torch
from models.wrappers.actor_observer_with_classifier_wrapper import ActorObserverWithClassifierWrapper


class ActorObserverClassifierWrapper(ActorObserverWithClassifierWrapper):
    def __init__(self, basenet, opts, *args, **kwargs):
        if 'DataParallel' in basenet.__class__.__name__:
            basenet = basenet.module
        print('Initializing classifier with AOWC instance')
        self.__dict__ = basenet.__dict__

    def forward(self, x, meta):
        base_x = self.basenet(x)
        y = self.classifier(base_x)
        w_x = self.firstpos_fc(base_x).view(-1) * torch.exp(self.firstpos_scale)
        w_z = self.firstneg_fc(base_x).view(-1) * torch.exp(self.firstneg_scale)
        print('fc7 norms: {}', base_x.data.norm())
        self.verbose()
        # return y, w_x, w_z
        return y
