#!/usr/bin/env python
import sys
import os
import subprocess
import traceback
import pdb
from bdb import BdbQuit
subprocess.Popen('find ./exp/.. -iname "*.pyc" -delete'.split())
sys.path.insert(0, '.')
os.nice(19)
from main import main
name = __file__.split('/')[-1].split('.')[0]  # name is filename

args = [
    '--name', name,
    '--dataset', 'charades_video_sfm2',
    '--print-freq', '1',
    '--arch', 'resnet18',
    '--wrapper', 'sfmlearner_wrapper',
    '--criterion', 'sfmlearner_criterion',
    '--train-file', '/nfs.yoda/gsigurds/CharadesEgo_v1_train.csv',
    '--val-file', '/nfs.yoda/gsigurds/CharadesEgo_v1_test.csv',
    '--data', '/scratch/gsigurds/CharadesEgo_v1_rgb/',
    '--resume', '/nfs.yoda/gsigurds/caches/' + name + '/model.pth.tar',
    '--lr', '2e-4',
    '--optimizer', 'adam',
    '--weight-decay', '0',
    '--batch-size', '10',
    '--train-size', '8.0',
    '--val-size', '1.0',
    '--cache-dir', '/nfs.yoda/gsigurds/caches/',
    '--epochs', '40',
    '--workers', '4',
    '--photo-loss-weight', '1.0',
    '--mask-loss-weight', '0.02',
    '--smooth-loss-weight', '0.1',
    '--inverse-loss-weight', '0',
    '--intrinsics-type', 'scaled',
    '--intrinsics-true-inv',
    '--metrics', '',
    '--metric', 'val_loss',
    '--tasks', 'depth_visualization_task',
    #'--evaluate',
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
