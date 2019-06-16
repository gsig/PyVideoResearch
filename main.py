#!/usr/bin/env python

"""Charades activity recognition baseline code
   Can be run directly or throught config scripts under exp/

   Gunnar Sigurdsson, 2018
"""
import torch
import torch.backends.cudnn as cudnn
import train
from models.get import get_model
from datasets.get import get_dataset
import checkpoints
from checkpoints import score_file
from opts import parse
from misc_utils import tee
from misc_utils.utils import seed
from misc_utils.experiments import get_script_dir_commit_hash, experiment_folder
from metrics.get import get_metrics
from tasks.get import get_tasks
import pdb
from bdb import BdbQuit
import traceback
import sys

# pytorch bugfixes
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def validate(trainer, val_loader, model, criterion, args, metrics, tasks, epoch=-1):
    scores = {}
    scores.update(trainer.validate(val_loader, model, criterion, epoch, metrics, args))
    for task in tasks:
        with torch.no_grad():
            scores.update(task.run(model, criterion, epoch, args))
    return scores


def main():
    best_score = 0
    args = parse()
    if not args.no_logger:
        tee.Tee(args.cache+'/log.txt')
    print(vars(args))
    print('experiment folder: {}'.format(experiment_folder()))
    print('git hash: {}'.format(get_script_dir_commit_hash()))
    seed(args.manual_seed)
    cudnn.benchmark = not args.disable_cudnn_benchmark
    cudnn.enabled = not args.disable_cudnn

    metrics = get_metrics(args.metrics)
    tasks = get_tasks(args.tasks)
    model, criterion = get_model(args)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                     weight_decay=args.weight_decay)
    else:
        assert False, "invalid optimizer"

    if args.resume:
        best_score = checkpoints.load(args, model, optimizer)
    print(model)
    trainer = train.Trainer()
    train_loader, val_loader = get_dataset(args)

    if args.evaluate:
        scores = validate(trainer, val_loader, model, criterion, args, metrics, tasks, -1)
        print(scores)
        score_file(scores, "{}/model_999.txt".format(args.cache))
        return

    if args.warmups > 0:
        for i in range(args.warmups):
            print('warmup {}'.format(i))
            trainer.validate(train_loader, model, criterion, -1, metrics, args)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            trainer.train_sampler.set_epoch(epoch)
        scores = {}
        scores.update(trainer.train(train_loader, model, criterion, optimizer, epoch, metrics, args))
        scores.update(validate(trainer, val_loader, model, criterion, args, metrics, tasks, epoch))
        is_best = scores[args.metric] > best_score
        best_score = max(scores[args.metric], best_score)
        checkpoints.save(epoch, args, model, optimizer, is_best, scores, args.metric)

         
def pdbmain():
    try:
        main()
    except BdbQuit:
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        print('')
        pdb.post_mortem()
        sys.exit(1)


if __name__ == '__main__':
    main()
