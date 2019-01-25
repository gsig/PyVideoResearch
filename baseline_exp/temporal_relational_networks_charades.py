#!/usr/bin/env python
# Temporal Relation Networks on top of ResNet152
# Trained on Charades
# original name: trn1f
# model_best.txt:
#     loss_train 5.97576176698
#     loss_val 6.22081581752
#     mAP 0.202797473287
#     top1train 33.0660324097
#     top1val 26.3888893127
#     top5train 120.11026001
#     top5val 100.0
#     videoprec1 28.5560932159
#     videoprec5 114.063339233
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
    '--dataset', 'charades_tsn',
    '--arch', 'resnet152',
    '--wrapper', 'trn_base',
    '--criterion', 'default',
    '--lr', '2.5e-4',
    '--lr-decay-rate', '60',
    '--epochs', '140',
    '--batch-size', '30',
    '--train-size', '1.0',
    '--weight-decay', '0.0000001',
    '--window-smooth', '0',
    '--val-size', '0.2',
    '--cache-dir', '/nfs.yoda/gsigurds/ai2/caches/',
    '--data', '/scratch/gsigurds/Charades_v1_rgb/',
    '--train-file', '/home/gsigurds/ai2/twostream/Charades_v1_train.csv',
    '--val-file', '/home/gsigurds/ai2/twostream/Charades_v1_test.csv',
    '--pretrained',
    '--balanceloss',
    '--nhidden', '3',
    '--originalloss-weight', '15',
    '--resume', '/nfs.yoda/gsigurds/ai2/caches/' + __file__.split('/')[-1].split('.')[0] + '/model.pth.tar',
    '--workers', '4',
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
