#!/usr/bin/env python
import sys
import os
import subprocess
import traceback
import pdb
from bdb import BdbQuit
subprocess.Popen('find ./exp/.. -iname "*.pyc" -delete'.split())
sys.path.insert(0, '.')
os.nice(19)
from main import main
name = __file__.split('/')[-1].split('.')[0]  # name is filename

args = [
    '--name', name,
    '--dataset', 'charades_ego',
    '--print-freq', '1',
    '--arch', 'resnet50',
    '--wrapper', 'actor_observer_wrapper',
    '--metrics', 'triplet_accuracy_metric;triplet_top5_metric;triplet_top10_metric;triplet_top50_metric',
    '--metric', 'val_triplet_top10_metric',
    '--criterion', 'actor_observer_criterion',
    '--tasks', 'alignment_task',
    '--train-file', '/nfs.yoda/gsigurds/CharadesEgo_v1_train.csv',
    '--val-file', '/nfs.yoda/gsigurds/CharadesEgo_v1_test.csv',
    '--data', '/scratch/gsigurds/CharadesEgo_v1_rgb/',
    '--pretrained',
    '--resume', '/nfs.yoda/gsigurds/caches/321a2/model_best.pth.tar',
    '--decay', '0.95',
    '--lr', '0e-5',
    '--weight-decay', '0',
    '--lr-decay-rate', '15',
    '--batch-size', '15',
    '--train-size', '0.2',
    '--val-size', '0.5',
    '--cache-dir', '/nfs.yoda/gsigurds/caches/',
    '--epochs', '50',
    '--workers', '4',
    '--normalize-per-video',
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
