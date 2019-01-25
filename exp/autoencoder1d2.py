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
    '--dataset', 'charades_video_x4',
    '--arch', 'resnet50_3d_autoencoder',
    '--lr', '0.005',
    '--lr-decay-rate', '25',
    '--wrapper', 'default',
    '--criterion', 'autoencoder_criterion',
    '--epochs', '75',
    '--batch-size', '6',
    #'--batch-size', '32',
    '--train-size', '4.0',
    '--weight-decay', '0.0000001',
    '--window-smooth', '0',
    '--val-size', '1.0',
    '--cache-dir', '/nfs.yoda/gsigurds/caches/',
    '--train-file', '/nfs.yoda/gsigurds/CharadesEgo_v1_train.csv',
    '--val-file', '/nfs.yoda/gsigurds/CharadesEgo_v1_test.csv',
    '--data', '/scratch/gsigurds/CharadesEgo_v1_rgb/',
    '--pretrained',
    '--nclass', '400',
    '--resume', '/nfs.yoda/gsigurds/caches/' + __file__.split('/')[-1].split('.')[0] + '/model.pth.tar',
    #'--workers', '16',
    '--workers', '6',
    '--metric', 'val_loss',
    '--metrics', '',
    '--tasks', 'visualization_task',
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
