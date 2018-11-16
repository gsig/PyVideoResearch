# pylint: disable=W0221,E1101
from __future__ import absolute_import
import torch
import torch.nn as nn
import sys
import os.path
from models.criteria.default_criterion import DefaultCriterion
from torch.nn import functional as F


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = torch.zeros(gt_loc.shape).cuda()
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    loc_loss /= ((gt_label >= 0).sum().float())
    return loc_loss


class FRCNNCriterion(DefaultCriterion):
    def __init__(self, args):
        super(FRCNNCriterion, self).__init__(args)
        del sys.modules['utils']
        del sys.modules['model']
        this_dir = os.path.dirname(__file__)
        lib_path = os.path.join(this_dir, '../wrappers/simple-faster-rcnn-pytorch')
        sys.path.insert(0, lib_path)
        from utils.array_tool import totensor, tonumpy
        from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator
        sys.path.pop(0)
        self.totensor = totensor
        self.tonumpy = tonumpy
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()
        self.img_size = args.input_size
        self.rpn_sigma = 3.
        self.roi_sigma = 1.
        self.nclass = args.nclass
        del sys.modules["utils"]

    def forward(self, score_prediction, roi_score, roi_cls_loc,
                rpn_scores, rpn_locs,
                anchor, gt_roi_loc, gt_roi_label,
                target, meta, synchronous=False):
        bbox = meta['box']
        if bbox.dim() == 3:
            bbox = bbox[0]
        bbox = bbox * self.img_size
        bbox = bbox[:, [1, 0, 3, 2]]

        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        img_size = (self.img_size, self.img_size)

        # ------------------ RPN losses -------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            self.tonumpy(bbox),
            anchor,
            img_size)
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            self.totensor(gt_rpn_loc).float(),
            self.totensor(gt_rpn_label).long().data,
            self.rpn_sigma)
        rpn_cls_loss = F.cross_entropy(rpn_score, self.totensor(gt_rpn_label).long().cuda(), ignore_index=-1)

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[torch.arange(0, n_sample).long().cuda(),
                              self.totensor(gt_roi_label).long()]
        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            self.totensor(gt_roi_loc).float(),
            self.totensor(gt_roi_label).long().data,
            self.roi_sigma)

        #roi_cls_loss = nn.CrossEntropyLoss()(roi_score, self.totensor(gt_roi_label).long().cuda())
        sigmoid_label = torch.zeros(gt_roi_label.shape[0], self.nclass)
        for ii, g in enumerate(gt_roi_label):
            if g > 0:  # labelled box
                for i, p in enumerate(meta['pids']):
                    if p == meta['pid']:  # same box
                        sigmoid_label[ii, meta['labels'][i]+1] = 1
            else:  # background box
                sigmoid_label[ii, 0] = 1
        if self.balance_loss and self.training:
            print('balancing loss')
            roi_score = self.balance_labels(roi_score, sigmoid_label)
        roi_cls_loss = nn.BCEWithLogitsLoss()(roi_score, sigmoid_label.cuda())

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        print('{} {} {} {}'.format(*losses))
        losses = losses + [sum(losses)]

        bbox = meta['boxes']
        if bbox.dim() == 3:
            bbox = bbox[0]
        label = torch.Tensor(meta['labels'])
        score_target = {'boxes': bbox.numpy(),
                        'labels': label.numpy(),
                        'start': meta['start'],
                        'vid': meta['id'][0]}

        # return a, loss, target
        return score_prediction, losses[-1], score_target
