#!/usr/bin/env python
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
    '--dataset', 'ava_mp4',
    '--arch', 'aj_i3d',
    '--wrapper', 'frcnn_wrapper3',
    '--criterion', 'frcnn_criterion3',
    '--metrics', 'frcnn_metric;frcnn_map_metric;frcnn_metric6',
    '--lr', '0.0025',
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
    '--originalloss-weight', '1',
    '--resume', '/nfs.yoda/gsigurds/caches/' + __file__.split('/')[-1].split('.')[0] + '/model.pth.tar' +
                ';/nfs.yoda/gsigurds/ai2/caches/avacls2b/model.pth.tar',
    '--workers', '4',
    '--no-val-video',
    '--metric', 'AVA6val',
    '--disable-cudnn-benchmark',
    '--freeze-batchnorm',
    '--freeze-base',
    '--freeze-head',
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
