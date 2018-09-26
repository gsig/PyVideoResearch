"""
Initialize the model module
New models can be defined by adding scripts under models/
"""
from importlib import import_module
from models.utils import set_distributed_backend, replace_last_layer, generic_load, case_getattr
from models.layers.default_criterion import DefaultCriterion


def get_model(args):
    """ Create base model, and wrap it with an optional wrapper, useful for extending models
    """

    model = generic_load(args.arch, args.pretrained, args.pretrained_weights, args)
    model = replace_last_layer(model, args)
    if not args.wrapper == 'default':
        #wrapper = case_getattr(import_module('.wrappers.' + args.wrapper, package='models.layers'), args.wrapper)
        wrapper = case_getattr(import_module('models.wrappers.' + args.wrapper), args.wrapper)
        model = wrapper(model, args)
    model = set_distributed_backend(model, args)

    # define loss function
    if args.criterion == 'default':
        criterion = DefaultCriterion
    else:
        #criterion = case_getattr(import_module('.criteria' + args.criterion, package='models.layers'), args.criterion)
        criterion = case_getattr(import_module('models.criteria.' + args.criterion), args.criterion)
    criterion = criterion(args).cuda()
    return model, criterion
