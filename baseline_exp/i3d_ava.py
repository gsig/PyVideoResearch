#!/usr/bin/env python
# i3d model trained on ava
# using a frame classification setup
# original name: avacls2b
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
    '--lr', '0.005',
    '--lr-decay-rate', '10',
    '--wrapper', 'default',
    '--criterion', 'background_criterion',
    '--epochs', '30',
    '--batch-size', '18',
    '--train-size', '1.0',
    '--weight-decay', '0.0000001',
    '--val-size', '0.1',
    '--cache-dir', '/nfs.yoda/gsigurds/caches/',
    '--data', '/scratch/gsigurds/processed_videos2/',
    '--train-file', '/nfs.yoda/gsigurds/ava/ava_train_v2.1.csv',
    '--val-file', '/nfs.yoda/gsigurds/ava/ava_val_v2.1.csv',
    '--pretrained',
    '--nclass', '81',
    '--resume', '/nfs.yoda/gsigurds/caches/' + name + '/model.pth.tar' +
                ';/nfs.yoda/gsigurds/charades_pretrained/aj_rgb_imagenet.pth',
    '--workers', '12',
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
