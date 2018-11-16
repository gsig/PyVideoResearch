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
    '--dataset', 'kinetics_mp4',
    #'--dataset', 'charades',
    #'--dataset', 'mock_dataset2',
    #'--arch', 'aj_i3dnlgrp',
    #'--arch', 'aj_i3d',
    #'--arch', 'aj_i3dnl',
    '--arch', 'aj_i3dnlgrpsafe',
    '--wrapper', 'default',
    '--criterion', 'default',
    '--lr', '2.5e-2',
    '--lr-decay-rate', '20',
    '--epochs', '50',
    '--memory-decay', '1.0',
    '--memory-size', '1',
    #'--input-size', '288',
    '--batch-size', '8',
    '--train-size', '0.1',
    '--temporal-weight', '0.000001',
    '--weight-decay', '0.0000001',
    '--temporalloss-weight', '0',
    '--window-smooth', '0',
    '--sigma', '300',
    '--val-size', '0.05',
    '--cache-dir', '/nfs.yoda/gsigurds/ai2/caches/',
    '--data', '/scratch/gsigurds/kinetics_compress/train_256/',
    '--valdata', '/scratch/gsigurds/kinetics_compress/val_256/',
    '--train-file', '/nfs.yoda/gsigurds/kinetics400/kinetics_train.csv',
    '--val-file', '/nfs.yoda/gsigurds/kinetics400/kinetics_val.csv',
    '--nclass', '400',
    #'--data', '/scratch/gsigurds/Charades_v1_rgb/',
    #'--train-file', '/home/gsigurds/ai2/twostream/Charades_v1_train.csv',
    #'--val-file', '/home/gsigurds/ai2/twostream/Charades_v1_test.csv',
    #'--nclass', '157',
    '--pretrained',
    #'--adjustment',
    '--balanceloss',
    '--nhidden', '3',
    #'--synchronous',
    '--originalloss-weight', '15',
    #'--videoloss',
    #'--resume', '/nfs.yoda/gsigurds/charades_pretrained/aj_rgb_imagenet.pth',
    #'--resume', '/nfs.yoda/gsigurds/ai2/caches/' + __file__.split('/')[-1].split('.')[0] + '/model.pth.tar'+';'+'/nfs.yoda/gsigurds/charades_pretrained/aj_rgb_imagenet.pth',
    #'--resume', '/nfs.yoda/gsigurds/ai2/caches/' + __file__.split('/')[-1].split('.')[0] + '/model.pth.tar'+';'+'/nfs.yoda/gsigurds/ai2/caches/i3d8d/model.pth.tar',
    #'--start-epoch', '2',
    '--resume', '/nfs.yoda/gsigurds/ai2/caches/' + __file__.split('/')[-1].split('.')[0] + '/model.pth.tar',
    #'--evaluate',
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
    print ''
    pdb.post_mortem()
    sys.exit(1)
