#!/usr/bin/env python
# Single frame baseline on Kinetics
# orignal name: frame1
# model_best.txt:
#     top1 33.3060836792
#     top1val 38.2186088562
#     top5 60.6615333557
#     top5val 65.1382141113
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
    '--dataset', 'kinetics',
    '--arch', 'resnet152',
    '--lr', '0.375',
    '--criterion', 'default',
    '--wrapper', 'default',
    '--lr-decay-rate', '8',
    '--epochs', '20',
    '--batch-size', '50',
    '--train-size', '0.1',
    '--val-size', '0.05',
    '--cache-dir', '/nfs.yoda/gsigurds/caches/',
    '--data', '/scratch/gsigurds/kinetics/',
    '--train-file', '/nfs.yoda/gsigurds/kinetics400/kinetics_train.csv',
    '--val-file', '/nfs.yoda/gsigurds/kinetics400/kinetics_val.csv',
    '--pretrained',
    '--nclass', '400',
    '--resume', '/nfs.yoda/gsigurds/ai2/caches/' + name + '/model.pth.tar',
    '--workers', '4',
    '--metric', 'val_top1',
    '--replace-last-layer'
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
