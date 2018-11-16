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
    '--dataset', 'something_something_tsn',
    '--arch', 'resnet152',
    '--wrapper', 'trn_base',
    #'--criterion', 'default',
    '--criterion', 'softmax_criterion',
    #'--lr', '2.5e-2',
    #'--lr', '3.75e-3',
    '--lr', '0.0125',
    '--lr-decay-rate', '50',
    '--temporal-segments', '7',
    '--epochs', '122',
    '--batch-size', '7',
    '--train-size', '0.3',
    '--weight-decay', '0.0001',
    '--window-smooth', '0',
    '--val-size', '0.2',
    '--cache-dir', '/nfs.yoda/gsigurds/ai2/caches/',
    '--data', '/scratch/gsigurds/20bn-something-something-v2/',
    '--train-file', '/nfs.yoda/gsigurds/somethingsomething/something-something-v2-train.json',
    '--val-file', '/nfs.yoda/gsigurds/somethingsomething/something-something-v2-validation.json',
    '--label-file', '/nfs.yoda/gsigurds/somethingsomething/something-something-v2-labels.json',
    '--pretrained',
    '--nclass', '174',
    #'--balanceloss',
    '--originalloss-weight', '1',
    '--resume', '/nfs.yoda/gsigurds/ai2/caches/' + __file__.split('/')[-1].split('.')[0] + '/model.pth.tar',
    #'--evaluate',
    '--workers', '4',
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
