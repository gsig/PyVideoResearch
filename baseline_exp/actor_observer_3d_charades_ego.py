#!/usr/bin/env python
# Actor Observer third to first person baseline
# using ResNet50 3D
# original name: 321bdebug
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
    '--dataset', 'charades_ego_video',
    '--print-freq', '1',
    '--arch', 'resnet50_3d',
    '--wrapper', 'actor_observer_wrapper',
    '--metrics', 'triplet_accuracy_metric;triplet_top5_metric;triplet_top10_metric;triplet_top50_metric',
    '--metric', 'val_triplet_top10_metric',
    '--tasks', 'alignment_3d_task',
    '--criterion', 'actor_observer_criterion',
    '--train-file', '/nfs.yoda/gsigurds/CharadesEgo_v1_train.csv',
    '--val-file', '/nfs.yoda/gsigurds/CharadesEgo_v1_test.csv',
    '--data', '/scratch/gsigurds/CharadesEgo_v1_rgb/',
    '--pretrained',
    '--resume', '/nfs.yoda/gsigurds/caches/' + name + '/model.pth.tar' +
                ';/nfs.yoda/gsigurds/caches/i3d12b2/model_best.pth.tar',
    '--start-epoch', '1',
    '--decay', '0.95',
    '--lr', '3e-5',
    '--lr-decay-rate', '15',
    '--batch-size', '3',
    '--video-batch-size', '16',
    '--train-size', '2.0',
    '--val-size', '1.0',
    '--cache-dir', '/nfs.yoda/gsigurds/caches/',
    '--epochs', '50',
    '--workers', '6',
    '--finaldecay', '0.9',
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
