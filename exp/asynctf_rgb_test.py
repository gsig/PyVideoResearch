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
    '--dataset', 'charadesrgb',
    '--arch', 'resnet152',
    '--lr', '.25e-2',
    '--lr-decay-rate', '3',
    '--epochs', '20',
    '--memory-decay', '1.0',
    '--memory-size', '20',
    '--batch-size', '50',
    '--train-size', '0.2',
    '--temporal-weight', '0.03',
    '--temporalloss-weight', '1.2',
    '--window-smooth', '1',
    '--sigma', '300',
    '--val-size', '0.001',
    '--cache-dir', '/nfs.yoda/gsigurds/ai2/caches/',
    '--data', '/glusterfs/gsigurds/Charades_v1_rgb/',
    '--pretrained',
    '--adjustment',
    '--balanceloss',
    '--nhidden', '3',
    '--originalloss-weight', '15',
    '--resume', 'asynctf_rgb.pth.tar',
    '--evaluate',
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
