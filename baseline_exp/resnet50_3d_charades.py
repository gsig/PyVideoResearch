#!/usr/bin/env python
# fine-tune Kinetics-pretrained 3D ResNet50 pretrained on Charades
# original name: i3d12b2
# model_best.txt:
#     CharadesmAPvalvideo 0.31270594963775783
#     loss_train 0.05916500749547305
#     loss_val 0.10518467018531787
#     top1train 42.33025029675106
#     top1val 39.729729729729726
#     top5train 121.67071984435681
#     top5val 133.51351351351352
#     videotop1valvideo 63.66076221148685
#     videotop5valvideo 231.07890499194846
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
    '--dataset', 'charades_video',
    '--arch', 'resnet50_3d',
    '--lr', '.375',
    '--criterion', 'default_criterion',
    '--wrapper', 'default_wrapper',
    '--lr-decay-rate', '15,40',
    '--epochs', '50',
    '--batch-size', '5',
    '--video-batch-size', '5',
    '--train-size', '1.0',
    '--weight-decay', '0.0000001',
    '--window-smooth', '0',
    '--val-size', '0.2',
    '--cache-dir', '/nfs.yoda/gsigurds/caches/',
    '--data', '/scratch/gsigurds/Charades_v1_rgb/',
    '--train-file', '/home/gsigurds/ai2/twostream/Charades_v1_train.csv',
    '--val-file', '/home/gsigurds/ai2/twostream/Charades_v1_test.csv',
    '--pretrained',
    '--resume', '/nfs.yoda/gsigurds/ai2/caches/i3d8k/model_best.pth.tar',
    '--start-epoch', '1',
    '--workers', '4',
    '--disable-cudnn-benchmark',
    '--disable-cudnn',
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
