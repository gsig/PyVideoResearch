import unittest
import train
import torch
from datasets.get import get_dataset
import torch.nn as nn
from models.layers.AsyncTFBase import AsyncTFBase
from models.layers.AsyncTFCriterion import AsyncTFCriterion
from opts import parse
import subprocess
subprocess.Popen('find ./exp/.. -iname "*.pyc" -delete'.split())


def opts(opt):
    opt.dataset = 'mock_dataset1'
    opt.lr_decay_rate = 100
    opt.lr = 1e-1
    opt.temporal_weight = 5.
    opt.temporalloss_weight = 0.05
    opt.memory_decay = 1.0
    opt.sigma = 150
    opt.print_freq = 9
    opt.weight_decay = 0
    opt.weight_decay = 5e-4
    opt.memory_size = 20
    opt.nclass = 5
    #opt.adjustment = True
    #opt.nhidden = 2
    opt.nhidden = 10


def simpletest1():
    # test if the code can learn a simple sequence
    opt = parse()
    opts(opt)
    epochs = 40
    train_loader, val_loader, valvideo_loader = get_dataset(opt)
    trainer = train.Trainer()
    basemodel = nn.Linear(100, 5)
    model = AsyncTFBase(basemodel, 100, opt).cuda()
    criterion = AsyncTFCriterion(opt).cuda()
    optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    epoch = -1
    for i in range(epochs):
        top1, _ = trainer.train(train_loader, model, criterion, optimizer, i, opt)
        print('cls weights: {}, aa weights: {}'.format(
            model.mA.parameters().next().norm().data[0],
            model.mAAa.parameters().next().norm().data[0]))
    top1, _ = trainer.validate(train_loader, model, criterion, epochs, opt)

    for i in range(5):
        top1val, _ = trainer.validate(val_loader, model, criterion, epochs + i, opt)
        print('top1val: {}'.format(top1val))

    ap = trainer.validate_video(valvideo_loader, model, criterion, epoch, opt)
    return top1, top1val, ap


class AsyncTests(unittest.TestCase):
    def test1(self):
        top1, top1val, ap = simpletest1()
        self.failUnless(top1 > 90)
        self.failUnless(top1val > 90)
        self.failUnless(ap > 0.9)

    #def test2(self):
    #    # stresstest
    #    top1s, top1vals, aps = [], [], []
    #    trials = 20
    #    for _ in range(trials):
    #        top1, top1val, ap = simpletest1()
    #        top1s.append(top1)
    #        top1vals.append(top1val)
    #        aps.append(ap)
    #    top1s = [1 if x > 85 else 0 for x in top1s]
    #    top1vals = [1 if x > 85 else 0 for x in top1vals]
    #    aps = [1 if x > .85 else 0 for x in aps]
    #    print('top1s: {}/{} \t top1vals: {}/{} \t aps: {}/{}'.format(sum(top1s), trials, sum(top1vals), trials, sum(aps), trials))
    #    self.failUnless(sum(top1s) > .8 * len(top1s))
    #    self.failUnless(sum(top1vals) > .8 * len(top1vals))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
