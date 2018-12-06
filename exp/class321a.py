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
    '--arch', 'resnet152',
    '--wrapper', 'actor_observer_with_classification_wrapper',
    '--metrics', 'triplet_accuracy_metric;triplet_top5_metric;triplet_top10_metric;triplet_top50_metric;top1_metric',
    '--tasks', 'alignment_task',
    '--metric', 'alignment_task_1sec',
    '--criterion', 'actor_observer_with_classification_criterion',
    '--train-file', '/nfs.yoda/gsigurds/CharadesEgo_v1_train.csv;/nfs.yoda/gsigurds/Charades_v1_train.csv',
    '--val-file', '/nfs.yoda/gsigurds/CharadesEgo_v1_test.csv;/nfs.yoda/gsigurds/Charades_v1_test.csv',
    '--data', '/scratch/gsigurds/CharadesEgo_v1_rgb/;/scratch/gsigurds/Charades_v1_rgb/',
    '--pretrained',
    '--resume', '/nfs.yoda/gsigurds/caches/' + name + '/model.pth.tar' +
                ';/nfs.yoda/gsigurds/charades_pretrained/resnet_rgb_python3.pth.tar',
    '--decay', '0.95',
    '--lr', '1e-5',
    '--classifier_weight', '2.0',
    '--lr-decay-rate', '8',
    '--batch-size', '15',
    '--train-size', str(0.2 / 6),
    '--val-size', '0.1',
    '--cache-dir', '/nfs.yoda/gsigurds/caches/',
    '--epochs', '50',
    '--workers', '4',
    '--no-val-video',
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
