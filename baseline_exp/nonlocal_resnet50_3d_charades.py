#!/usr/bin/env python
# Non-local 3D ResNet50 
# pretrained on Kinetics
# fine-tuning on Charades
# original name: i3d13b
# model_best.txt:
#     CharadesmAPvalvideo 0.31509522635404147
#     loss_train 0.05229745948463118
#     loss_val 0.11164817365037429
#     top1train 51.6221341564131
#     top1val 46.75675675675676
#     top5train 145.78099624764053
#     top5val 151.35135135135135
#     videotop1valvideo 63.016639828234034
#     videotop5valvideo 234.99731615673645
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
    '--dataset', 'charades_video',
    '--arch', 'resnet50_3d_nonlocal',
    '--lr', '.375',
    '--criterion', 'default_criterion',
    '--wrapper', 'default_wrapper',
    '--lr-decay-rate', '15,40',
    '--epochs', '100',
    '--batch-size', '5',
    '--video-batch-size', '5',
    '--train-size', '1.0',
    '--weight-decay', '0.0000001',
    '--val-size', '0.2',
    '--cache-dir', '/nfs.yoda/gsigurds/caches/',
    '--data', '/scratch/gsigurds/Charades_v1_rgb/',
    '--train-file', '/nfs.yoda/gsigurds/Charades_v1_train.csv',
    '--val-file', '/nfs.yoda/gsigurds/Charades_v1_test.csv',
    '--pretrained',
    '--start-epoch', '1',
    '--resume', '/nfs.yoda/gsigurds/caches/' + name + '/model.pth.tar' +
                ';/nfs.yoda/gsigurds/ai2/caches/i3d8l/model_best.pth.tar',
    '--workers', '4',
    '--disable-cudnn-benchmark',
    '--disable-cudnn',
    '--tasks', 'video_task',
    '--replace-last-layer',
    '--metric', 'video_task_CharadesmAP',
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
