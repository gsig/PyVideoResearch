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
    '--dataset', 'charades_video_x',
    '--arch', 'resnet50_3d',
    '--criterion', 'default',
    '--wrapper', 'default',
    #'--criterion', 'async_tf_criterion',
    #'--wrapper', 'async_tf_base',
    '--lr', '0.0006',
    '--lr-decay-rate', '50',
    '--epochs', '50',
    #'--memory-decay', '1.0',
    #'--memory-size', '20',
    '--batch-size', '12',
    '--video-batch-size', '12',
    '--train-size', '5.0',
    '--input-size', '256',
    #'--input-size', '224',
    #'--temporal-weight', '0.03',
    '--weight-decay', '0.0001',
    #'--temporalloss-weight', '1.2',
    '--window-smooth', '0',
    #'--sigma', '300',
    #'--val-size', '0.2',
    '--val-size', '0.005',
    '--cache-dir', '/nfs.yoda/gsigurds/ai2/caches/',
    '--data', '/scratch/gsigurds/Charades_v1_rgb/',
    '--train-file', '/home/gsigurds/ai2/twostream/Charades_v1_train.csv',
    '--val-file', '/home/gsigurds/ai2/twostream/Charades_v1_test.csv',
    '--pretrained',
    #'--adjustment',
    #'--balanceloss',
    #'--warmups', '10',
    #'--nhidden', '3',
    #'--synchronous',
    '--originalloss-weight', '1',
    #'--videoloss',
    #'--resume', '/nfs.yoda/gsigurds/charades_pretrained/aj_rgb_imagenet.pth',
    '--resume', '/nfs.yoda/gsigurds/ai2/caches/' + __file__.split('/')[-1].split('.')[0] + '/model.pth.tar'+';'+'/nfs.yoda/gsigurds/ai2/caches/i3d8k/model_best.pth.tar',
    '--start-epoch', '1',
    #'--evaluate',
    '--workers', '10',
    #'--workers', '0',
    '--disable-cudnn-benchmark',
    '--disable-cudnn',
    '--freeze-batchnorm',
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
