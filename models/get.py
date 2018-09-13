"""
Initialize the model module
New models can be defined by adding scripts under models/
"""
import importlib
from models.utils import set_distributed_backend, replace_last_layer, generic_load
from models.layers.DefaultCriterion import DefaultCriterion


def create_model(args):
    """ Create base model, and wrap it with an optional wrapper, useful for extending models
    """

    if args.wrapper == 'default':
        wrapper = lambda x, y, z: x
    else:
        wrapper = getattr(importlib.import_module('.' + args.wrapper, package='models.layers'), args.wrapper)

    model = generic_load(args.arch, args.pretrained, args.pretrained_weights)
    model = replace_last_layer(model, args)
    model = wrapper(model, model.in_features, args)
    model = set_distributed_backend(model, args)

    # define loss function
    if args.criterion == 'default':
        criterion = DefaultCriterion
    else:
        criterion = getattr(importlib.import_module('.' + args.criterion, package='models.layers'), args.criterion)
    criterion = criterion(args).cuda()
    return model, criterion
