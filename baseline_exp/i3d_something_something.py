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
import subprocess
subprocess.Popen('find ./exp/.. -iname "*.pyc" -delete'.split())

args = [
    '--name', __file__.split('/')[-1].split('.')[0],  # name is filename
    '--print-freq', '1',
    '--dataset', 'something_something_webm',
    '--arch', 'aj_i3d',
    '--lr', '2.5e-2',
    '--lr-decay-rate', '20',
    '--epochs', '50',
    '--memory-decay', '1.0',
    '--memory-size', '1',
    '--batch-size', '5',
    '--train-size', '0.1',
    '--temporal-weight', '0.000001',
    '--weight-decay', '0.0000001',
    '--temporalloss-weight', '0',
    '--window-smooth', '0',
    '--sigma', '300',
    '--val-size', '0.05',
    '--cache-dir', '/nfs.yoda/gsigurds/ai2/caches/',
    '--data', '/scratch/gsigurds/20bn-something-something-v2/',
    '--train-file', '/nfs.yoda/gsigurds/somethingsomething/something-something-v2-train.json',
    '--val-file', '/nfs.yoda/gsigurds/somethingsomething/something-something-v2-validation.json',
    '--label-file', '/nfs.yoda/gsigurds/somethingsomething/something-something-v2-labels.json',
    '--pretrained',
    '--nclass', '174',
    '--balanceloss',
    '--nhidden', '3',
    '--originalloss-weight', '15',
    '--resume', '/nfs.yoda/gsigurds/ai2/caches/' + __file__.split('/')[-1].split('.')[0] + '/model.pth.tar'+';'+'/nfs.yoda/gsigurds/charades_pretrained/aj_rgb_imagenet.pth',
    '--workers', '6',
    '--no-val-video',
    '--metric', 'top1val',
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
