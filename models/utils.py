import torchvision.models as tmodels
from importlib import import_module
import torch
import torch.nn as nn
import torch.distributed as dist
import collections


class IdentityModule(nn.Module):
    def forward(self, inputs):
        return inputs


def split_list(lst, chunk_num):
    n = len(lst)
    chunk_size = max(1, n // chunk_num)
    i = 0
    out_lst = []
    while i < n:
        out_lst.append(lst[i:i+chunk_size])
        i += chunk_size
    return tuple(out_lst)


class MyDataParallel(nn.DataParallel):
    # Overloads nn.DataParallel to provide the ability to skip
    # scatter/gather functionality for a simple unmodified
    # list of dictionaries
    def __init__(self, *args, **kwargs):
        super(MyDataParallel, self).__init__(*args, **kwargs)

    def scatter(self, inputs, kwargs, device_ids):
        # only scatter inputs that don't have the do_not_collate flag
        inputss = []
        for inp in inputs:
            if isinstance(inp, collections.Sequence):
                if isinstance(inp[0], collections.Mapping) and 'do_not_collate' in inp[0]:
                    inp = split_list(inp, len(device_ids))
                else:
                    inp, kwargs = super(MyDataParallel, self).scatter((inp, ), kwargs, device_ids)
                    inp = [x[0] for x in inp]  # de-tuple
            else:
                inp, kwargs = super(MyDataParallel, self).scatter((inp, ), kwargs, device_ids)
                inp = [x[0] for x in inp]  # de-tuple
                kwargs = [x[0] for x in kwargs]  # de-tuple
            inputss.append(inp)
        return tuple(zip(*inputss)), kwargs

    def gather(self, outputs, output_device):
        # only gather outputs that don't have the do_not_collate flag
        # return should be #args x #batch
        outputss = []
        if isinstance(outputs[0], tuple):
            # multiple output arguments
            # outputs is #gpu x #args x #gpu_batch
            for out in zip(*outputs):  # out is #gpu x #gpu_batch
                if (isinstance(out[0], collections.Sequence) and
                   isinstance(out[0][0], collections.Mapping) and
                   'do_not_collate' in out[0][0]):
                    out = [x for y in out for x in y]  # join lists
                else:
                    out = super(MyDataParallel, self).gather(out, output_device)
                outputss.append(out)
            return tuple(outputss)
        else:
            # outputs is #gpu x #gpu_batch
            return super(MyDataParallel, self).gather(outputs, output_device)


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
        model = case_getattr(import_module('models.bases.' + arch), arch).get(args)
        if not weights == '':
            print('loading pretrained-weights from {}'.format(weights))
            model.load_state_dict(torch.load(weights))
    return model


def replace_last_layer(model, args):
    if hasattr(model, 'replace_logits'):
        model.replace_logits(args.nclass)
    elif hasattr(model, 'classifier'):
        newcls = list(model.classifier.children())
        model.classifier = nn.Sequential(*newcls[:-1])
    elif hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, args.nclass)
        if hasattr(model, 'AuxLogits'):
            model.AuxLogits.fc = nn.Linear(model.AuxLogists.fc.in_features, args.nclass)
    else:
        newcls = list(model.children())[:-1]
        model = nn.Sequential(*newcls)
    return model


def remove_last_layer(model):
    if hasattr(model, 'classifier'):
        newcls = list(model.classifier.children())
        model.in_features = newcls[-1].in_features
        model.classifier = nn.Sequential(*newcls[:-1])
    elif hasattr(model, 'fc'):
        if isinstance(model.fc, nn.Conv3d):
            model.in_features = model.fc.in_channels
            model.fc = IdentityModule()
            model.global_pooling = True
        else:
            model.in_features = model.fc.in_features
            model.fc = IdentityModule()
    else:
        newcls = list(model.children())[:-1]
        model.in_features = list(model.children())[-1].in_features
        model = nn.Sequential(*newcls[:-1])
    return model


def set_distributed_backend(model, args):
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if hasattr(model, 'features'):
            model.features = MyDataParallel(model.features)
            model.cuda()
        else:
            model = MyDataParallel(model).cuda()
    return model
