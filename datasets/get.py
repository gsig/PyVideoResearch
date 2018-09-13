""" Initilize the datasets module
    New datasets can be added with python scripts under datasets/
"""
import torch
import torch.utils.data
import torch.utils.data.distributed
import importlib


def case_getattr(obj, attr):
    casemap = {}
    for x in obj.__dict__:
        casemap[x.lower()] = x
    return getattr(obj, casemap[attr.lower()])


def get_dataset(args):
    dataset = importlib.import_module('.' + args.dataset.lower(), package='datasets')
    datasets = case_getattr(dataset, args.dataset).get(args)
    train_dataset, val_dataset, valvideo_dataset = datasets[:3]

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(
            train_sampler is None) and not args.synchronous,
        num_workers=args.workers, pin_memory=False, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=not args.synchronous,
        num_workers=args.workers, pin_memory=False)

    valvideo_loader = torch.utils.data.DataLoader(
        valvideo_dataset, batch_size=valvideo_dataset.testGAP, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    return train_loader, val_loader, valvideo_loader
