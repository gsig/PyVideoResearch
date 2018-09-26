import torchvision.models as tmodels
from importlib import import_module
import torch
import torch.nn as nn
import torch.distributed as dist


def case_getattr(obj, attr):
    casemap = {}
    for x in obj.__dict__:
        casemap[x.lower().replace('_', '')] = x
    return getattr(obj, casemap[attr.lower().replace('_', '')])


def generic_load(arch, pretrained, weights, args):
    if arch in tmodels.__dict__:  # torchvision models
        if pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            model = tmodels.__dict__[arch](pretrained=True)
            model = model.cuda()
        else:
            print("=> creating model '{}'".format(arch))
            model = tmodels.__dict__[arch]()
    else:  # defined as script in bases
        #model = import_module('.bases.' + arch, package='models').get(args)
        model = case_getattr(import_module('models.bases.' + arch), arch).get(args)
        if not weights == '':
            print('loading pretrained-weights from {}'.format(weights))
            model.load_state_dict(torch.load(weights))
    return model


def replace_last_layer(model, args):
    if hasattr(model, 'classifier'):
        newcls = list(model.classifier.children())
        model.classifier = nn.Sequential(*newcls[:-1])
    elif hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, args.nclass)
        if hasattr(model, 'AuxLogits'):
            model.AuxLogits.fc = nn.Linear(model.AuxLogists.fc.in_features, args.nclass)
    elif hasattr(model, 'replace_logits'):
        model.replace_logits(args.nclass)
    else:
        newcls = list(model.children())[:-1]
        model = nn.Sequential(*newcls)
    return model


def set_distributed_backend(model, args):
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if hasattr(model, 'features'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    return model
