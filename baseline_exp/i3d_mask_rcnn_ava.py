#!/usr/bin/env python
# Mask-RCNN baseline trained on AVA
# original name: ava10e
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
    '--dataset', 'ava_mp4',
    '--arch', 'aj_i3d',
    '--wrapper', 'maskrcnn_wrapper',
    '--criterion', 'maskrcnn_criterion',
    '--metrics', 'frcnn_metric6',
    '--lr', '0.1',
    '--lr-decay-rate', '50',
    '--input-size', '400',
    '--epochs', '120',
    '--batch-size', '8',
    '--train-size', '1.0',
    '--weight-decay', '0.0001',
    '--val-size', '1.0',
    '--cache-dir', '/nfs.yoda/gsigurds/caches/',
    '--data', '/scratch/gsigurds/processed_videos2/',
    '--train-file', '/nfs.yoda/gsigurds/ava/ava_train_v2.1.csv',
    '--val-file', '/nfs.yoda/gsigurds/ava/ava_val_v2.1.csv',
    '--pretrained',
    '--nclass', '81',
    '--resume', '/nfs.yoda/gsigurds/caches/' + name + '/model.pth.tar',
    '--workers', '12',
    '--metric', 'val_AVA6',
    '--disable-cudnn-benchmark',
    '--freeze-batchnorm',
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
