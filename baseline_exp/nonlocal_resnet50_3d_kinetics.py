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
import pdb
import traceback
sys.path.insert(0, '.')
from main import main
from bdb import BdbQuit
import os
os.nice(19)
import subprocess
subprocess.Popen('find ./exp/.. -iname "*.pyc" -delete'.split())

args = [
    '--name', __file__.split('/')[-1].split('.')[0],  # name is filename
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
    '--window-smooth', '0',
    '--val-size', '0.1',
    '--cache-dir', '/nfs.yoda/gsigurds/ai2/caches/',
    '--data', '/scratch/gsigurds/kinetics_compress/train_256/',
    '--valdata', '/scratch/gsigurds/kinetics_compress/val_256/',
    '--train-file', '/nfs.yoda/gsigurds/kinetics400/kinetics_train.csv',
    '--val-file', '/nfs.yoda/gsigurds/kinetics400/kinetics_val.csv',
    '--pretrained',
    '--nclass', '400',
    '--originalloss-weight', '1.',
    '--resume', '/nfs.yoda/gsigurds/ai2/caches/' + __file__.split('/')[-1].split('.')[0] + '/model.pth.tar',
    '--workers', '16',
    '--no-val-video',
    '--metric', 'top1val',
    '--disable-cudnn-benchmark',
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
