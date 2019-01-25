""" Defines functions used for checkpointing models and storing model scores """
import os
import torch
import shutil
from collections import OrderedDict


def ordered_load_state(model, chkpoint):
    """
        Wrapping the model with parallel/dataparallel seems to
        change the variable names for the states
        This attempts to load normally and otherwise aligns the labels
        of the two states and tries again.
    """
    try:
        model.load_state_dict(chkpoint)
    except RuntimeError as e:  # assume order is the same, and use new labels
        print(e)
        print('keys do not match model, trying to align')
        model_keys = model.state_dict().keys()
        fixed = OrderedDict([(z, y) for (_, y), z in zip(chkpoint.items(), model_keys)])
        model.load_state_dict(fixed)


def load_partial_state(model, state_dict):
    # @chenyuntc
    sd = model.state_dict()
    sd = OrderedDict([(x.replace('module.', '').replace('mA.', '').replace('basenet.', '').replace('encoder.', ''), y) for x, y in sd.items()])
    for k0, v in state_dict.items():
        k = k0.replace('module.', '').replace('mA.', '').replace('basenet.', '').replace('encoder.', '')
        if k not in sd or not sd[k].shape == v.shape:
            print('ignoring state key for loading: {}'.format(k))
            continue
        if isinstance(v, torch.nn.Parameter):
            v = v.data
        sd[k].copy_(v)


def load(args, model, optimizer):
    if args.resume:
        for resume in args.resume.split(';'):
            if os.path.isfile(resume):
                print("=> loading checkpoint '{}'".format(resume))
                chkpoint = torch.load(resume)
                if isinstance(chkpoint, dict) and 'state_dict' in chkpoint:
                    try:
                        ordered_load_state(model, chkpoint['state_dict'])
                        optimizer.load_state_dict(chkpoint['optimizer'])
                    except Exception as e:
                        print(e)
                        print('loading partial state 2')
                        load_partial_state(model, chkpoint['state_dict'])
                    print("=> loaded checkpoint '{}' (epoch {})"
                          .format(resume, chkpoint['epoch']))
                    if args.start_epoch == 0:
                        args.start_epoch = chkpoint['epoch']
                        print('setting start epoch to model epoch {}'.format(args.start_epoch))
                    if 'scores' in chkpoint and args.metric in chkpoint['scores']:
                        best_metric = chkpoint['scores'][args.metric]
                    else:
                        best_metric = 0
                    return best_metric
                else:
                    try:
                        ordered_load_state(model, chkpoint)
                    except Exception as e:
                        print(e)
                        print('loading partial state')
                        load_partial_state(model, chkpoint)
                    print("=> loaded checkpoint '{}' (just weights)".format(resume))
                    return 0
                break
            else:
                print("=> no checkpoint found, starting from scratch: '{}'".format(resume))
    return 0


def score_file(scores, filename):
    with open(filename, 'w') as f:
        for key, val in sorted(scores.items()):
            f.write('{} {}\n'.format(key, val))


def save(epoch, args, model, optimizer, is_best, scores, metric):
    state = {
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'best_metric': scores[metric],
        'scores': scores,
        'optimizer': optimizer.state_dict(),
    }
    filename = "{}/model.pth.tar".format(args.cache)
    score_file(scores, "{}/model_{:03d}.txt".format(args.cache, epoch+1))
    torch.save(state, filename)
    if is_best:
        bestname = "{}/model_best.pth.tar".format(args.cache)
        score_file(scores, "{}/model_best.txt".format(args.cache, epoch+1))
        shutil.copyfile(filename, bestname)
