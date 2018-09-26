import torch
import time
import numpy as np
import random


def seed(manualseed):
    random.seed(manualseed)
    np.random.seed(manualseed)
    torch.manual_seed(manualseed)
    torch.cuda.manual_seed(manualseed)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.dim() == 3:
        target = target.max(dim=1)[0]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    if len(target.shape) == 1:
        print('computing accuracy for single-label case')
        correct = pred.eq(target.view(1, -1).expand_as(pred))
    else:
        print('computing accuracy for multi-label case')
        correct = torch.zeros(*pred.shape)
        for i in range(correct.shape[0]):
            for j in range(correct.shape[1]):
                correct[i, j] = target[j, pred[i, j]] > 0.5

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def submission_file(ids, outputs, filename):
    """ write list of ids and outputs to filename"""
    with open(filename, 'w') as f:
        for vid, output in zip(ids, outputs):
            scores = ['{:g}'.format(x)
                      for x in output]
            f.write('{} {}\n'.format(vid, ' '.join(scores)))


class Timer(AverageMeter):
    def __init__(self):
        super(Timer, self).__init__()
        self.end = time.time()

    def tic(self):
        self.update(time.time() - self.end)
        self.end = time.time()

    def thetime(self):
        return time.time()
