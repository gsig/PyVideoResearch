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
    '--dataset', 'charades',
    '--arch', 'resnet152',
    #'--wrapper', 'default',
    #'--criterion', 'default',
    #'--temporal-segments', '1',
    '--lr', '2.5e-3',
    '--lr-decay-rate', '3',
    '--epochs', '10',
    '--batch-size', '50',
    #'--train-size', '20.0',
    '--train-size', '0.2',
    '--weight-decay', '0.0001',
    '--window-smooth', '1',
    '--memory-size', '20',
    '--sigma', '300',
    '--temporal-weight', '0.03',
    '--temporalloss-weight', '1.2',
    '--val-size', '0.1',
    '--cache-dir', '/nfs.yoda/gsigurds/ai2/caches/',
    '--data', '/scratch/gsigurds/Charades_v1_rgb/',
    '--train-file', '/home/gsigurds/ai2/twostream/Charades_v1_train.csv',
    '--val-file', '/home/gsigurds/ai2/twostream/Charades_v1_test.csv',
    '--pretrained',
    '--adjustment',
    #'--balanceloss',
    '--nhidden', '3',
    '--originalloss-weight', '15',
    #'--resume', '/nfs.yoda/gsigurds/ai2/caches/' + __file__.split('/')[-1].split('.')[0] + '/model.pth.tar',
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
    print ''
    pdb.post_mortem()
    sys.exit(1)
