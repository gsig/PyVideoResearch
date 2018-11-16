"""
Initialize the model module
New models can be defined by adding scripts under models/
"""
from importlib import import_module
from models.utils import set_distributed_backend, replace_last_layer, generic_load, case_getattr
import torch.nn


def get_model(args):
    """ Create base model, and wrap it with an optional wrapper, useful for extending models
    """

    model = generic_load(args.arch, args.pretrained, args.pretrained_weights, args)
    model = replace_last_layer(model, args)
    for module in model.modules():
        if args.dropout != 0 and isinstance(module, torch.nn.modules.Dropout):
            print('setting Dropout p to {}'.format(args.dropout))
            module.p = args.dropout

    wrapper = case_getattr(import_module('models.wrappers.' + args.wrapper), args.wrapper)
    model = wrapper(model, args)
    model = set_distributed_backend(model, args)

    # define loss function
    criterion = case_getattr(import_module('models.criteria.' + args.criterion), args.criterion)
    criterion = criterion(args).cuda()
    return model, criterion
