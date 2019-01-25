"""
   Use ActorObserver model as first person classifier
"""
import torch
from models.wrappers.actor_observer_wrapper import ActorObserverWrapper


class ActorObserverFC7Wrapper(ActorObserverWrapper):
    def __init__(self, basenet, opts, *args, **kwargs):
        if 'DataParallel' in basenet.__class__.__name__:
            basenet = basenet.module
        print('Initializing FC7 extractor with AOB instance')
        self.__dict__ = basenet.__dict__

    def forward(self, x, y, z):
        """ assuming:
            x: first person positive
            y: third person
            z: first person negative
        """
        base_x = self.basenet(x)
        base_y = self.basenet(y)
        w_x = self.firstpos_fc(base_x).view(-1) * torch.exp(self.firstpos_scale)
        w_y = self.third_fc(base_x).view(-1) * torch.exp(self.third_scale)
        print('fc7 norms:', base_x.norm().item(), base_y.norm().item())
        self.verbose()
        return base_x, base_y, w_x, w_y
