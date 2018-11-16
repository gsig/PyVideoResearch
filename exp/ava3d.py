#!/usr/bin/env python
import sys
import pdb
import traceback
#sys.path.insert(0, '..')
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
    '--wrapper', 'frcnn_wrapper2',
    '--criterion', 'frcnn_criterion2',
    '--metrics', 'frcnn_metric;frcnn_metric3;frcnn_map_metric;frcnn_metric6',
    '--lr', '0.001',
    '--lr-decay-rate', '50',
    '--input-size', '400',
    '--epochs', '120',
    '--batch-size', '4',
    '--train-size', '0.1',
    '--weight-decay', '0.0001',
    '--val-size', '0.1',
    '--cache-dir', '/nfs.yoda/gsigurds/ai2/caches/',
    '--data', '/scratch/gsigurds/processed_videos2/',
    '--train-file', '/nfs.yoda/gsigurds/ava/ava_train_v2.1.csv',
    '--val-file', '/nfs.yoda/gsigurds/ava/ava_val_v2.1.csv',
    '--pretrained',
    '--nclass', '81',
    '--originalloss-weight', '1',
    #'--resume', '/nfs.yoda/gsigurds/ai2/caches/' + __file__.split('/')[-1].split('.')[0] + '/model.pth.tar' + ';/nfs.yoda/gsigurds/charades_pretrained/aj_rgb_imagenet.pth',
    '--resume', '/nfs.yoda/gsigurds/charades_pretrained/aj_rgb_imagenet.pth',
    #'--evaluate',
    '--workers', '6',
    '--no-val-video',
    '--metric', 'AVA6val',
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
