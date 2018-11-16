# pylint: disable=W0221,E1101
from __future__ import absolute_import
import sys
import os.path
from models.criteria.default_criterion import DefaultCriterion


class FRCNNCriterion2(DefaultCriterion):
    def __init__(self, args):
        super(FRCNNCriterion2, self).__init__(args)
        this_dir = os.path.dirname(__file__)
        lib_path = os.path.join(this_dir, '../../external/Detectron.pytorch/')
        sys.path.insert(0, lib_path)
        from roi_data.rpn import add_rpn_blobs
        from modeling import rpn_heads, fast_rcnn_heads
        self.add_rpn_blobs = add_rpn_blobs
        self.generic_rpn_losses = rpn_heads.generic_rpn_losses
        self.fast_rcnn_losses = fast_rcnn_heads.fast_rcnn_losses
        self.img_size = args.input_size
        self.nclass = args.nclass

    def forward(self, score_prediction, roi_score, bbox_pred, rpn_ret, rpn_kwargs,
                target, meta, synchronous=False):
        # rpn loss
        loss_rpn_cls, loss_rpn_bbox = self.generic_rpn_losses(**rpn_kwargs)
        if loss_rpn_cls > 10 or loss_rpn_bbox > 10 or loss_rpn_cls != loss_rpn_cls:
            import pdb
            pdb.set_trace()

        # bbox loss
        print('roi_score norm {} \t mean {}'.format(roi_score.norm(), roi_score.mean()))
        if self.training:
            print(rpn_ret['labels_int32'])
            #rpn_ret['bbox_targets'] = np.tile(rpn_ret['bbox_targets'][:, -4:], (1, roi_score.shape[1]))
            #rpn_ret['bbox_inside_weights'] = np.tile(rpn_ret['bbox_inside_weights'][:, -4:], (1, roi_score.shape[1]))
            #rpn_ret['bbox_outside_weights'] = np.tile(rpn_ret['bbox_outside_weights'][:, -4:], (1, roi_score.shape[1]))
            loss_cls, loss_bbox, accuracy_cls = self.fast_rcnn_losses(
                roi_score, bbox_pred, rpn_ret['labels_int32'].cpu().numpy(), rpn_ret['bbox_targets'].cpu().numpy(),
                rpn_ret['bbox_inside_weights'].cpu().numpy(), rpn_ret['bbox_outside_weights'].cpu().numpy())
        else:
            loss_cls = loss_bbox = accuracy_cls = 0

        losses = [loss_rpn_cls, loss_rpn_bbox*2, loss_cls, loss_bbox*2]
        #losses = [loss_rpn_cls*0, loss_rpn_bbox*0, loss_cls, loss_bbox*0]
        print('losses {} {} {} {}'.format(*losses))
        print('accuracy {}'.format(accuracy_cls))

        score_target = []
        for m in meta:
            score_target.append({'boxes': m['boxes'],
                                 'labels': m['labels'],
                                 'start': m['start'],
                                 'vid': m['id']})

        return score_prediction, sum(losses), score_target
