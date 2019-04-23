#!/usr/bin/env python
# Nonlocal 3D ResNet50 evaluating on whole video on Kinetics using fully convolutional testing
# using 4 GPUs
# orignal name: nonlocal_resnet50_3d_kinetics_fullyconvtest
# model_best.txt:
#     videotop1 69.48361675653256
#     videotop5 89.54790543343012
import sys
import pdb
import traceback
sys.path.insert(0, '.')
from main import main
from bdb import BdbQuit
import os
os.nice(19)
name = __file__.split('/')[-1].split('.')[0]

args = [
    '--name', name,  # name is filename
    '--print-freq', '1',
    '--dataset', 'kinetics_mp4_x2',
    '--arch', 'resnet50_3d_nonlocal',
    '--lr', '0.005',
    '--lr-decay-rate', '100',
    '--wrapper', 'default',
    '--criterion', 'softmax_criterion',
    '--epochs', '300',
    '--batch-size', '1',
    '--train-size', '0.2',
    '--weight-decay', '0.0001',
    '--val-size', '0.0',
    '--cache-dir', '/nfs.yoda/gsigurds/caches/',
    '--data', '/scratch/gsigurds/kinetics_compress/train_256/',
    '--valdata', '/scratch/gsigurds/kinetics_compress/val_256/',
    '--train-file', '/nfs.yoda/gsigurds/kinetics400/kinetics_train.csv',
    '--val-file', '/nfs.yoda/gsigurds/kinetics400/kinetics_val.csv',
    '--pretrained',
    '--nclass', '400',
    '--resume', '/nfs.yoda/gsigurds/pretrained/i3d8l.pth.tar',
    '--workers', '16',
    '--metric', 'val_top1',
    '--video-metrics', 'videotop1_metric;videotop5_metric',
    '--tasks', 'video_task',
    '--disable-cudnn-benchmark',
    '--evaluate',
]
sys.argv.extend(args)
try:
    main()
except BdbQuit:
    sys.exit(1)
except Exception:
    traceback.print_exc()
    print('')
    pdb.post_mortem()
    sys.exit(1)
