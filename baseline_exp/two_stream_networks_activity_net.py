#!/usr/bin/env python
# Single frame ActivityNet model
# original name: anet2
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
    '--dataset', 'activitynet2',
    '--arch', 'resnet152',
    '--wrapper', 'default',
    '--criterion', 'default',
    '--lr', '0.01',
    '--lr-decay-rate', '50',
    '--epochs', '120',
    '--batch-size', '50',
    '--train-size', '1.0',
    '--weight-decay', '0.0001',
    '--val-size', '0.1',
    '--cache-dir', '/nfs.yoda/gsigurds/caches/',
    '--data', '/scratch/gsigurds/activitynet_jpg4/',
    '--train-file', '/nfs.yoda/gsigurds/activity_net.v1-3.min.json',
    '--val-file', '/nfs.yoda/gsigurds/activity_net.v1-3.min.json',
    '--label-file', '/nfs.yoda/gsigurds/activity_net.v1-3.min.json',
    '--nclass', '200',
    '--pretrained',
    '--resume', '/nfs.yoda/gsigurds/caches/' + name + '/model.pth.tar',
    '--workers', '4',
    '--replace-last-layer',
    '--tasks', 'video_task',
    '--metric', 'video_task_videotop1',
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
