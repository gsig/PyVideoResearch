#!/usr/bin/env python
# Actor Observer third to first person baseline
# single-frame model (ResNet152)
# orignal name: 321debug3
import sys
import os
import traceback
import pdb
from bdb import BdbQuit
sys.path.insert(0, '.')
os.nice(19)
from main import main
name = __file__.split('/')[-1].split('.')[0]  # name is filename

args = [
    '--name', name,
    '--dataset', 'charades_ego',
    '--print-freq', '1',
    '--arch', 'resnet152',
    '--wrapper', 'actor_observer_wrapper',
    '--metrics', 'triplet_accuracy_metric;triplet_top5_metric;triplet_top10_metric;triplet_top50_metric',
    '--metric', 'val_triplet_top10_metric',
    '--criterion', 'actor_observer_criterion',
    '--tasks', 'alignment_task',
    '--train-file', '/nfs.yoda/gsigurds/CharadesEgo_v1_train.csv',
    '--val-file', '/nfs.yoda/gsigurds/CharadesEgo_v1_test.csv',
    '--data', '/scratch/gsigurds/CharadesEgo_v1_rgb/',
    '--pretrained',
    '--resume', '/nfs.yoda/gsigurds/caches/' + name + '/model.pth.tar' +
                ';/nfs.yoda/gsigurds/charades_pretrained/resnet_rgb_python3.pth.tar',
    '--decay', '0.95',
    '--lr', '3e-5',
    '--lr-decay-rate', '15',
    '--batch-size', '15',
    '--train-size', '0.2',
    '--val-size', '0.5',
    '--cache-dir', '/nfs.yoda/gsigurds/caches/',
    '--epochs', '50',
    '--workers', '4',
    '--share-selector', 'False',
    '--finaldecay', '0.9',
    #'--evaluate',
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
