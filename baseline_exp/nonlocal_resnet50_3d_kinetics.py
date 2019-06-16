#!/usr/bin/env python
# Nonlocal 3D ResNet50 trained from scratch on Kinetics
# using 4 GPUs
# orignal name: i3d8l
# model_best.txt:
#     loss_train 1.3284634162060722
#     loss_val 1.4899196803569794
#     top1train 67.17897339699863
#     top1val 65.15625
#     top5train 86.47467598908595
#     top5val 85.57291666666667
import sys
sys.path.insert(0, '.')
from main import pdbmain
import os
os.nice(19)
name = __file__.split('/')[-1].split('.')[0]

args = [
    '--name', name,  # name is filename
    '--print-freq', '1',
    '--dataset', 'kinetics_mp4_x',
    '--arch', 'resnet50_3d_nonlocal',
    '--lr', '0.005',
    '--lr-decay-rate', '100',
    '--wrapper', 'default',
    '--criterion', 'softmax_criterion',
    '--epochs', '300',
    '--batch-size', '32',
    '--train-size', '0.2',
    '--weight-decay', '0.0001',
    '--val-size', '0.1',
    '--cache-dir', '/nfs.yoda/gsigurds/caches/',
    '--data', '/scratch/gsigurds/kinetics_compress/train_256/',
    '--valdata', '/scratch/gsigurds/kinetics_compress/val_256/',
    '--train-file', '/nfs.yoda/gsigurds/kinetics400/kinetics_train.csv',
    '--val-file', '/nfs.yoda/gsigurds/kinetics400/kinetics_val.csv',
    '--pretrained',
    '--nclass', '400',
    '--resume', '/nfs.yoda/gsigurds/caches/' + name + '/model.pth.tar',
    '--workers', '16',
    '--metric', 'val_top1',
    '--disable-cudnn-benchmark',
]
sys.argv.extend(args)
pdbmain()
