""" Initilize the datasets module
    New datasets can be added with python scripts under datasets/
"""
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
import torch.utils.data.distributed
import importlib
import collections


def case_getattr(obj, attr):
    casemap = {}
    for x in obj.__dict__:
        casemap[x.lower()] = x
    return getattr(obj, casemap[attr.replace('_', '')])


def cat_collate(batch):
    # the dataset returns a list, which gets wrapped in a list, we just unwrap the list
    # and feed it to the original dataloader
    assert len(batch) == 1, 'something wrong with val video dataset'
    return default_collate(batch[0])


def my_collate(batch):
    if isinstance(batch[0], collections.Mapping) and 'do_not_collate' in batch[0]:
        return batch
    if isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [my_collate(samples) for samples in transposed]
    else:
        return default_collate(batch)


def get_dataset(args):
    obj = importlib.import_module('.' + args.dataset, package='datasets')
    datasets = case_getattr(obj, args.dataset).get(args)
    train_dataset, val_dataset, valvideo_dataset = datasets[:3]
    print(train_dataset)
    print(val_dataset)
    print(valvideo_dataset)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, collate_fn=my_collate, shuffle=(
            train_sampler is None) and not args.synchronous, drop_last=True,
        num_workers=args.workers, pin_memory=False, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, collate_fn=my_collate, drop_last=True,
        shuffle=not args.synchronous, num_workers=args.workers, pin_memory=False)

    valvideo_loader = torch.utils.data.DataLoader(
        valvideo_dataset, batch_size=1, shuffle=False, collate_fn=cat_collate,
        num_workers=args.workers, pin_memory=False)

    return train_loader, val_loader, valvideo_loader