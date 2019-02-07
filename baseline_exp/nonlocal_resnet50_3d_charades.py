#!/usr/bin/env python
# Non-local 3D ResNet50 
# pretrained on Kinetics
# fine-tuning on Charades
# original name: i3d31b
# model_best.txt:
#     train_loss 0.058912872455846095
#     train_top1 43.606517911256105
#     train_top5 125.22190584455218
#     val_loss 0.10237216038836373
#     val_top1 42.22222222222222
#     val_top5 147.77777777777777
#     video_task_CharadesmAP 0.32087015304465993
#     video_task_videotop1 64.30488459473966
#     video_task_videotop5 237.30542136339238
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
                ';/nfs.yoda/gsigurds/ai2/caches/i3d8l2/model_best.pth.tar',
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
