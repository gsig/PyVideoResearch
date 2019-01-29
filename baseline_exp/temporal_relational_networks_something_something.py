#!/usr/bin/env python
# Temporal Relation Network trained on Something Something
# original name: trn4b
# model_best.txt:
#     CharadesmAPvalvideo 0.170306601953
#     loss_train 1.24716269897
#     loss_val 3.23484053724
#     top1train 64.8668923753
#     top1val 40.4156579527
#     top5train 89.0355817911
#     top5val 69.3099282381
#     videotop1valvideo 41.9986277596
#     videotop5valvideo 72.5309763087
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
    '--dataset', 'something_something_tsn',
    '--arch', 'resnet152',
    '--wrapper', 'trn_base',
    '--criterion', 'softmax_criterion',
    '--lr', '0.00125',
    '--lr-decay-rate', '50',
    '--temporal-segments', '7',
    '--epochs', '122',
    '--batch-size', '7',
    '--train-size', '0.3',
    '--weight-decay', '0.0001',
    '--val-size', '0.2',
    '--cache-dir', '/nfs.yoda/gsigurds/caches/',
    '--data', '/scratch/gsigurds/20bn-something-something-v2/',
    '--train-file', '/nfs.yoda/gsigurds/somethingsomething/something-something-v2-train.json',
    '--val-file', '/nfs.yoda/gsigurds/somethingsomething/something-something-v2-validation.json',
    '--label-file', '/nfs.yoda/gsigurds/somethingsomething/something-something-v2-labels.json',
    '--pretrained',
    '--nclass', '174',
    '--resume', '/nfs.yoda/gsigurds/caches/' + name + '/model.pth.tar',
    '--workers', '4',
    '--metric', 'val_top1',
    '--replace-last-layer',
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
