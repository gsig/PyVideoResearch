# pylint: disable=W0221,E1101
from __future__ import absolute_import
import sys
import os.path
import torch
from models.criteria.default_criterion import DefaultCriterion


class MaskRCNNCriterion(DefaultCriterion):
    def __init__(self, args):
        super(MaskRCNNCriterion, self).__init__(args)

        # initialize Detectron
        this_dir = os.path.dirname(__file__)
        lib_path = os.path.join(this_dir, '../../external/maskrcnn-benchmark')
        sys.path.insert(0, lib_path)
        from maskrcnn_benchmark.config import cfg
        self.cfg = cfg
        self.img_size = args.input_size
        self.nclass = args.nclass
        self.sigmoid = True

    def forward(self, score_predictions, proposal_losses, detector_losses,
                target, meta, synchronous=False):

        if self.training:
            if proposal_losses['loss_objectness'].dim() == 0:
                losses = [proposal_losses['loss_objectness'],
                          proposal_losses['loss_rpn_box_reg'],
                          detector_losses['loss_classifier'],
                          detector_losses['loss_box_reg'],
                          ]
            else:
                losses = [sum(proposal_losses['loss_objectness']),
                          sum(proposal_losses['loss_rpn_box_reg']),
                          sum(detector_losses['loss_classifier']),
                          sum(detector_losses['loss_box_reg']),
                          ]

            print('losses {} {} {} {}'.format(*losses))
        else:
            losses = [torch.zeros(1)]

        score_targets = []
        for m in meta:
            score_targets.append({'boxes': m['boxes'],
                                  'labels': m['labels'],
                                  'start': m['start'],
                                  'vid': m['id']})

        return score_predictions, sum(losses), score_targets
