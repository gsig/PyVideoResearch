#!/usr/bin/env python

"""Charades activity recognition baseline code
   Can be run directly or throught config scripts under exp/

   Gunnar Sigurdsson, 2018
"""
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random
import train
from models.get import create_model
from datasets.get import get_dataset
import checkpoints
from opts import parse
from utils import tee
from utils.experiments import get_script_dir_commit_hash, experiment_checksums, experiment_folder


def seed(manualseed):
    random.seed(manualseed)
    np.random.seed(manualseed)
    torch.manual_seed(manualseed)
    torch.cuda.manual_seed(manualseed)


def validate(trainer, val_loader, valvideo_loader, model, criterion, args, epoch=-1):
    scores = {}
    scores.update(trainer.validate(val_loader, model, criterion, epoch, args))
    if not args.no_val_video:
        scores.update(trainer.validate_video(valvideo_loader, model, criterion, epoch, args))


def main():
    best_score = 0
    args = parse()
    if not args.no_logger:
        tee.Tee(args.cache+'/log.txt')
    print(vars(args))
    print('experiment folder: {}'.format(experiment_folder()))
    print('git hash: {}'.format(get_script_dir_commit_hash()))
    print('checksums:\n{}'.format(experiment_checksums()))
    seed(args.manual_seed)

    model, criterion = create_model(args)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    cudnn.benchmark = True

    if args.resume:
        best_score = checkpoints.load(args, model, optimizer)
    print(model)
    trainer = train.Trainer()
    train_loader, val_loader, valvideo_loader = get_dataset(args)

    if args.evaluate:
        trainer.validate(val_loader, model, criterion, -1, args)
        trainer.validate_video(valvideo_loader, model, criterion, -1, args)
        return

    if args.warmups > 0:
        for i in range(args.warmups):
            print('warmup {}'.format(i))
            trainer.validate(train_loader, model, criterion, -1, args)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            trainer.train_sampler.set_epoch(epoch)
        scores = {}
        scores.update(trainer.train(train_loader, model, criterion, optimizer, epoch, args))
        is_best = scores[args.metric] > best_score
        best_score = max(scores[args.metric], best_score)
        checkpoints.save(epoch, args, model, optimizer, is_best, scores, args.metric)


if __name__ == '__main__':
    main()
