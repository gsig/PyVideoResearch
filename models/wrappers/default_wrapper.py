from models.wrappers.wrapper import Wrapper
import torch


class DefaultWrapper(Wrapper):
    def __init__(self, basenet, opts, *args, **kwargs):
        super(DefaultWrapper, self).__init__(basenet, opts, *args, **kwargs)
        self.freeze_batchnorm = opts.freeze_batchnorm

    def forward(self, x, meta):
        if self.freeze_batchnorm:
            for module in self.basenet.modules():
                if isinstance(module, torch.nn.modules.BatchNorm1d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm2d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm3d):
                    module.eval()
        return self.basenet(x)
