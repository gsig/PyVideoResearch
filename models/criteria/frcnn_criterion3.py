# pylint: disable=W0221,E1101
from __future__ import absolute_import
import sys
import os.path
from models.criteria.default_criterion import DefaultCriterion
import torch.nn


class FRCNNCriterion3(DefaultCriterion):
    def __init__(self, args):
        super(FRCNNCriterion3, self).__init__(args)

        # initialize Detectron
        this_dir = os.path.dirname(__file__)
        lib_path = os.path.join(this_dir, '../../external/Detectron.pytorch/')
        sys.path.insert(0, lib_path)
        from roi_data.rpn import add_rpn_blobs
        from modeling import rpn_heads, fast_rcnn_heads
        from core.config import cfg
        self.cfg = cfg
        self.add_rpn_blobs = add_rpn_blobs
        self.generic_rpn_losses = rpn_heads.generic_rpn_losses
        self.fast_rcnn_losses = fast_rcnn_heads.fast_rcnn_losses
        self.img_size = args.input_size
        self.nclass = args.nclass
        self.sigmoid = True

    def forward(self, score_predictions, roi_score, bbox_pred, rpn_ret, rpn_kwargs,
                target, meta, synchronous=False):

        # rpn loss
        loss_rpn_cls, loss_rpn_bbox = self.generic_rpn_losses(**rpn_kwargs)

        # bbox loss
        print('roi_score norm {} \t mean {}'.format(roi_score.norm(), roi_score.mean()))
        if self.training:
            if self.cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
                labels = rpn_ret['fix_labels_int32'].cpu().numpy() * rpn_ret['labels_int32'].cpu().numpy()
            else:
                labels = rpn_ret['labels_int32'].cpu().numpy()
            loss_cls, loss_bbox, accuracy_cls = self.fast_rcnn_losses(
                roi_score, bbox_pred, labels, rpn_ret['bbox_targets'].cpu().numpy(),
                rpn_ret['bbox_inside_weights'].cpu().numpy(), rpn_ret['bbox_outside_weights'].cpu().numpy())
        else:
            loss_cls = loss_bbox = accuracy_cls = 0

        if self.sigmoid:
            sigmoid_loss = torch.nn.MultiLabelSoftMarginLoss()
            loss_cls = sigmoid_loss(roi_score, rpn_ret['multilabels_int32'])
            print('sigmoid frcnn loss')

        losses = [loss_rpn_cls, loss_rpn_bbox*2, loss_cls*0, loss_bbox*0]
        print('losses {} {} {} {}'.format(*losses))
        print('accuracy {}'.format(accuracy_cls))

        score_targets = []
        for m in meta:
            score_targets.append({'boxes': m['boxes'],
                                  'labels': m['labels'],
                                  'start': m['start'],
                                  'vid': m['id']})

        return score_predictions, sum(losses), score_targets
