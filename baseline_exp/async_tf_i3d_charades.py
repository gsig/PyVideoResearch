#!/usr/bin/env python
# Asynchronous Temporal Fields model on top of a i3d model
# Trained on Charades
# original name: async_par1
# best model performance:
#     loss_train 0.375729223538
#     loss_val 1.700613914
#     mAP 0.336631962176
#     top1train 43.8626289368
#     top1val 43.2795715332
#     top5train 124.912445068
#     top5val 135.48387146
#     videoprec1 64.1438522339
#     videoprec5 236.392913818
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
    '--dataset', 'charades_video_x2',
    '--arch', 'aj_i3d',
    '--criterion', 'async_tf_criterion',
    '--wrapper', 'async_tf_wrapper',
    '--lr', '2.5e-3',
    '--lr-decay-rate', '8',
    '--epochs', '20',
    '--memory-decay', '1.0',
    '--memory-size', '20',
    '--batch-size', '12',
    '--video-batch-size', '10',
    '--input-size', '288', 
    '--train-size', '1.0',
    '--temporal-weight', '0.03',
    '--weight-decay', '0.0000001',
    '--temporalloss-weight', '1.2',
    '--window-smooth', '1',
    '--sigma', '300',
    '--val-size', '0.2',
    '--cache-dir', '/nfs.yoda/gsigurds/caches/',
    '--data', '/scratch/gsigurds/Charades_v1_rgb/',
    '--train-file', '/nfs.yoda/gsigurds/Charades_v1_train.csv',
    '--val-file', '/nfs.yoda/gsigurds/Charades_v1_test.csv',
    '--pretrained',
    '--balanceloss',
    '--nhidden', '3',
    '--originalloss-weight', '15',
    '--resume', '/nfs.yoda/gsigurds/caches/' + name + '/model.pth.tar' +
                ';/nfs.yoda/gsigurds/charades_pretrained/aj_rgb_charades.pth',
    '--workers', '4',
    '--tasks', 'video_task',
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
