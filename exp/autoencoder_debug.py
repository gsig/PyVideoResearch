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
    '--dataset', 'charades_video',
    '--print-freq', '1',
    '--arch', 'resnet50_3d_autoencoder4',
    '--wrapper', 'default',
    '--tasks', 'autoencoder_task',
    '--train-file', '/nfs.yoda/gsigurds/CharadesEgo_v1_train.csv',
    '--val-file', '/nfs.yoda/gsigurds/CharadesEgo_v1_test.csv',
    '--data', '/scratch/gsigurds/CharadesEgo_v1_rgb/',
    '--pretrained',
    #'--resume', '/nfs.yoda/gsigurds/caches/autoencoder1/model.pth.tar',
    '--lr', '1e-2',
    '--weight-decay', '1e-4',
    '--batch-size', '1',
    '--val-size', '0',
    '--cache-dir', '/nfs.yoda/gsigurds/caches/',
    '--epochs', '10000',
    '--workers', '0',
    '--evaluate',
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
