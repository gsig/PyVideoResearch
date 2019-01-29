#!/usr/bin/env python
# Kinetics-pretrained I3D fine-tuned on Something Something
# original name: i3d9b
# model_best.txt:
#     loss_train 0.149611573413
#     loss_val 0.387080801161
#     top1train 45.745010376
#     top1val 42.753036499
#     top5train 76.912727356
#     top5val 74.7368392944
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
    '--dataset', 'something_something_webm',
    '--arch', 'aj_i3d',
    '--lr', '0.375',
    '--lr-decay-rate', '20',
    '--epochs', '50',
    '--batch-size', '5',
    '--train-size', '0.1',
    '--weight-decay', '0.0000001',
    '--val-size', '0.05',
    '--cache-dir', '/nfs.yoda/gsigurds/caches/',
    '--data', '/scratch/gsigurds/20bn-something-something-v2/',
    '--train-file', '/nfs.yoda/gsigurds/somethingsomething/something-something-v2-train.json',
    '--val-file', '/nfs.yoda/gsigurds/somethingsomething/something-something-v2-validation.json',
    '--label-file', '/nfs.yoda/gsigurds/somethingsomething/something-something-v2-labels.json',
    '--pretrained',
    '--nclass', '174',
    '--resume', '/nfs.yoda/gsigurds/ai2/caches/' + name + '/model.pth.tar' +
                ';/nfs.yoda/gsigurds/charades_pretrained/aj_rgb_imagenet.pth',
    '--workers', '6',
    '--replace-last-layer',
    '--metric', 'val_top1',
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
