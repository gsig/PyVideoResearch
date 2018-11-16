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
    '--dataset', 'kinetics_mp4_x',
    '--arch', 'resnet101_3d',
    #'--arch', 'resnet50_3d',
    #'--arch', 'aj_i3d',
    #'--lr', '2.5e-2',
    #'--lr', '0.01',
    '--lr', '0.005',
    #'--lr', '1.0',
    #'--lr', '0.00125',
    #'--accum-grad', '8',
    #'--lr', '0.1',
    '--lr-decay-rate', '100',
    '--wrapper', 'default',
    '--criterion', 'softmax_criterion',
    '--epochs', '300',
    #'--input-size', '288',
    #'--batch-size', '20',
    #'--batch-size', '8',
    '--batch-size', '32',
    '--train-size', '0.2',
    '--weight-decay', '0.0000001',
    '--window-smooth', '0',
    '--val-size', '0.1',
    '--cache-dir', '/nfs.yoda/gsigurds/ai2/caches/',
    '--data', '/scratch/gsigurds/kinetics_compress/train_256/',
    '--valdata', '/scratch/gsigurds/kinetics_compress/val_256/',
    '--train-file', '/nfs.yoda/gsigurds/kinetics400/kinetics_train.csv',
    '--val-file', '/nfs.yoda/gsigurds/kinetics400/kinetics_val.csv',
    '--pretrained',
    '--nclass', '400',
    #'--adjustment',
    #'--balanceloss',
    #'--synchronous',
    '--originalloss-weight', '1.',
    #'--videoloss',
    #'--resume', '/nfs.yoda/gsigurds/charades_pretrained/aj_rgb_imagenet.pth',
    #'--resume', '/nfs.yoda/gsigurds/ai2/caches/' + __file__.split('/')[-1].split('.')[0] + '/model.pth.tar'+';'+'/nfs.yoda/gsigurds/charades_pretrained/aj_rgb_imagenet.pth',
    '--resume', '/nfs.yoda/gsigurds/ai2/caches/' + __file__.split('/')[-1].split('.')[0] + '/model.pth.tar',
    #'--evaluate',
    '--workers', '16',
    #'--workers', '0',
    #'--workers', '6',
    '--no-val-video',
    '--metric', 'top1val',
    '--disable-cudnn-benchmark',
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
