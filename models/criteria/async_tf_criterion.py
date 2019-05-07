# pylint: disable=W0221,E1101
import torch
import torch.nn as nn
import math
from random import random
from models.layers.verbose_gradients import VerboseGradients
# from memory_profiler import profile
from models.layers.utils import axb, avg
from models.criteria.utils import winsmooth
from models.criteria.default_criterion import DefaultCriterion


class MessagePassing(object):
    # Class for keeping track of messages across frames
    def __init__(self, maxsize, w_time, decay, sigma):
        super(MessagePassing, self).__init__()
        self.maxsize = maxsize
        self.w_time = w_time
        self.decay = decay
        self.sigma = sigma
        self.storage = {}
        self.storage_gt = {}
        self.training = self.training if hasattr(self, 'training') else True
        self.nc = None

    def mget(self, idtime, size, storage, cond=lambda t, t0: True, kernel=lambda t, t0: 1):
        # get message using condition on the timestamps
        def meta(ids, t0):
            try:
                return avg(((y, kernel(t, t0)) for t, y in storage[ids]
                            if cond(t, t0)), 1. / self.decay)
            except (StopIteration, KeyError):
                return torch.zeros(size)
        out = [meta(ids, time) for ids, time in idtime]
        return torch.stack(out, 0).cuda()

    def get_msg(self, idtime, time='past', storage=None):
        storage = self.storage if storage is None else storage
        cond = lambda t, t0: t < t0 if time == 'past' else t > t0
        kernel = lambda t, t0: math.exp(-float(t - t0)**2 / (2 * self.sigma**2))
        return self.mget(idtime, self.nc, storage, cond, kernel) * self.w_time

    def get_gt_msg(self, idtime, time='past'):
        return self.get_msg(idtime, time, self.storage_gt)

    def mset(self, msg, idtime, storage, mask):
        # keep a queue of size maxsize for each id
        # messages are stored in normal space
        # queue for each id is stored in the order in which the messages were stored
        for m, (ids, time), keep in sorted(zip(msg, idtime, mask), key=lambda x: random()):
            if not keep:
                continue
            if ids not in storage:
                storage[ids] = []
            data = m if type(m) is not torch.Tensor else m.data.cpu()
            storage[ids].append((time, data))
            if len(storage[ids]) > self.maxsize:
                del storage[ids][0]

    def set_msg(self, qa, idtime, mask):
        self.mset(qa, idtime, self.storage, mask)

    def set_gt_msg(self, qa, target, idtime, mask):
        x = target.data.cpu()
        self.mset(x, idtime, self.storage_gt, mask)


class AsyncTFCriterion(DefaultCriterion, MessagePassing):
    def __init__(self, args):
        MessagePassing.__init__(self, args.memory_size, args.temporal_weight, args.memory_decay, args.sigma)
        DefaultCriterion.__init__(self, args)
        self.msg_n = 5
        self.w_tloss = args.temporalloss_weight
        self.adjustment = args.adjustment

    def forward(self, a, aa, target, meta, niter=1, synchronous=False):
        mask = [True] * a.shape[0]
        # idtime = zip(meta['id'], meta['time'])
        idtime = [(m['id'], m['time']) for m in meta]  # for 'do_not_collate'
        if a.dim() == 3 and self.training:
            # temporal mode
            mask = [True if i == 0 else False for x in idtime for i in range(target.shape[1])]
            idtime_video = idtime
            idtime = [x for x in idtime for _ in range(target.shape[1])]
            idtime = idtime + idtime_video
        a, target, meta = self.process_tensors(a, target, meta)
        self.nc = a.shape[1]
        if aa.shape[0] < a.shape[0]:
            aa = aa[0].expand(a.shape[0], aa.shape[1], aa.shape[2])

        a, aa = VerboseGradients.apply(a, aa)
        msg = self.get_msg(idtime, 'past')
        fmsg = self.get_msg(idtime, 'future')
        qa = a.clone()
        qa += (aa * msg[:, :, None]).sum(1)
        qa += (aa * fmsg[:, None, :]).sum(2)
        qa = torch.nn.Sigmoid()(qa)
        if self.balance_loss and self.training:
            print('balancing loss')
            qa = self.balance_labels(qa, target)

        loss = self.loss(qa, target)
        loss += self.loss(torch.nn.Sigmoid()(a), target)
        # self.set_msg(a, idtime)
        # self.set_msg(qa, idtime)
        self.set_msg(nn.Sigmoid()(a), idtime, mask)

        if self.training:
            if self.adjustment:
                # This is an adjustment that makes the objective a true fully-connected CRF
                # Can be thought of as a regularizer on aa
                print('adding asynctf adjustment to loss')
                self.set_gt_msg(qa, target, idtime, mask)
                gt_msg = self.get_gt_msg(idtime, time='past')
                gt_fmsg = self.get_gt_msg(idtime, time='future')
                pastloss = .5 * axb(gt_msg - msg, aa, target).pow(2).mean() * self.w_tloss / self.nc**2
                futureloss = .5 * axb(target, aa, gt_fmsg - fmsg).pow(2).mean() * self.w_tloss / self.nc**2
            else:
                pastloss = loss * 0
                futureloss = loss * 0

            print('losses class: {} \t past: {} \t future: {}'.format(
                  loss.item(), pastloss.item(), futureloss.item()))
            loss = (loss + pastloss + futureloss) / 3

        if not synchronous or niter > self.msg_n:
            out = qa.clone()
            if synchronous:
                out = winsmooth(out, kernelsize=self.win_smooth)
            return out, loss, target
        else:
            return self.forward(a, aa, target, meta, niter=niter + 1, synchronous=synchronous)
