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
from opts import parse
from misc_utils import tee
from misc_utils.utils import seed
from misc_utils.experiments import get_script_dir_commit_hash, experiment_checksums, experiment_folder
from metrics.get import get_metrics

# pytorch bugfixes
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def validate(trainer, val_loader, valvideo_loader, model, criterion, args, metrics, videometrics, epoch=-1):
    scores = {}
    if not args.no_val_video:
        scores.update(trainer.validate_video(valvideo_loader, model, criterion, epoch, videometrics, args))
    scores.update(trainer.validate(val_loader, model, criterion, epoch, metrics, args))
    return scores


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
    cudnn.benchmark = not args.disable_cudnn_benchmark
    cudnn.enabled = not args.disable_cudnn

    metrics = get_metrics(args.metrics)
    videometrics = get_metrics(args.videometrics)
    model, criterion = get_model(args)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        best_score = checkpoints.load(args, model, optimizer)
    print(model)
    trainer = train.Trainer()
    train_loader, val_loader, valvideo_loader = get_dataset(args)

    if args.evaluate:
        validate(trainer, val_loader, valvideo_loader, model, criterion, args, metrics, videometrics, -1)
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
        scores.update(validate(trainer, val_loader, valvideo_loader, model, criterion, args, metrics, videometrics, epoch))
        is_best = scores[args.metric] > best_score
        best_score = max(scores[args.metric], best_score)
        checkpoints.save(epoch, args, model, optimizer, is_best, scores, args.metric)


if __name__ == '__main__':
    main()