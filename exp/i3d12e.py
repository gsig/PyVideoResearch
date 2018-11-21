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
    '--dataset', 'charades_video',
    '--arch', 'resnet50_3d',
    '--lr', '.375',
    '--criterion', 'default_criterion',
    '--wrapper', 'default_wrapper',
    '--lr-decay-rate', '15,40',
    '--epochs', '50',
    '--batch-size', '5',
    '--video-batch-size', '5',
    '--train-size', '1.0',
    '--weight-decay', '0.0001',
    '--window-smooth', '0',
    '--val-size', '0.2',
    '--cache-dir', '/nfs.yoda/gsigurds/caches/',
    '--data', '/scratch/gsigurds/Charades_v1_rgb/',
    '--train-file', '/home/gsigurds/ai2/twostream/Charades_v1_train.csv',
    '--val-file', '/home/gsigurds/ai2/twostream/Charades_v1_test.csv',
    '--pretrained',
    '--resume', '/nfs.yoda/gsigurds/ai2/caches/i3d8k/model_best.pth.tar',
    '--start-epoch', '1',
    #'--resume', '/nfs.yoda/gsigurds/caches/' + __file__.split('/')[-1].split('.')[0] + '/model.pth.tar',
    #'--evaluate',
    '--workers', '4',
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
