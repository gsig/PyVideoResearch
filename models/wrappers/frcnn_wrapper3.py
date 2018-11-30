from __future__ import absolute_import

from models.wrappers.wrapper import Wrapper
import torch
import sys
import os.path
import os
import scipy
import numpy as np
from models.bases.aj_i3d import Unit3D
from collections import defaultdict
import torch.nn.functional as F
import torch.nn.init as init


def self_swap(fun, self):
    def newfun(*args, **kwargs):
        return fun(self, *args, **kwargs)
    return newfun


def tensorize(x):
    for k, v in x.items():
        if not type(v) is torch.Tensor:
            x[k] = torch.Tensor(v).cuda().detach()


def str2torch(str_array):
    return torch.Tensor(np.fromstring(str_array, dtype=np.uint8))


def meta_to_entry(meta, nclasses, inputsize):
    entry = {}
    num_boxes = len(meta['labels'])
    entry['image'] = meta['id']
    entry['height'] = inputsize
    entry['width'] = inputsize
    entry['flipped'] = False
    entry['boxes'] = meta['boxes'].cpu().numpy() * inputsize
    entry['segms'] = []
    entry['seg_areas'] = np.empty((0), dtype=np.float32)
    entry['gt_classes'] = meta['labels'].cpu().numpy() + 1  # 0 is bg class
    entry['is_crowd'] = np.array([0]*num_boxes, dtype=np.bool)
    gt_overlaps = np.zeros((num_boxes, nclasses), dtype=np.float32)
    gt_overlaps[:, entry['gt_classes']] = 1
    entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps, dtype=np.float32)
    entry['box_to_gt_ind_map'] = np.where(entry['gt_classes'] > 0)[0]
    return entry


class FRCNNWrapper3(Wrapper):
    def init_detectron(self, args):
        if 'datasets' in sys.modules:
            del sys.modules['datasets']
        this_dir = os.path.dirname(__file__)
        lib_path = os.path.join(this_dir, '../../external/Detectron.pytorch/lib')
        sys.path.insert(0, lib_path)

        # config
        self.spatial_scale = 1. / 16
        self.dim_out = 832

        from core.config import cfg
        self.cfg = cfg

        cfg.MODEL.FASTER_RCNN = True
        cfg.MODEL.NUM_CLASSES = args.nclass
        cfg.MODEL.CLS_AGNOSTIC_BBOX_REG = True
        cfg.MODEL.FASTER_RCNN = True
        cfg.MODEL.TYPE = "generalized_rcnn"

        cfg.RPN.CLS_ACTIVATION = 'sigmoid'
        cfg.RPN.SIZES = (32, 64, 128, 256, 512)
        cfg.RPN.STRIDE = 16
        cfg.FPN.COARSEST_STRIDE = 16  # bugfix, anchor stride depends on FPN parameters

        cfg.TRAIN.BATCH_SIZE_PER_IM = 30
        cfg.TRAIN.IMS_PER_BATCH = 2
        cfg.TRAIN.MAX_SIZE = 400
        cfg.TRAIN.RPN_BATCH_SIZE_PER_IM = 256

        cfg.TEST.RPN_POST_NMS_TOP_N = 30
        cfg.TEST.DETECTIONS_PER_IM = 0  # turn off image cap (for rare classes)
        cfg.TEST.SCORE_THRESH = 0  # include all bounding boxes (for rare classes)

        cfg.FAST_RCNN.ROI_XFORM_METHOD = "RoIAlign"
        cfg.FAST_RCNN.ROI_XFORM_RESOLUTION = 7
        cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO = 2

        # bind classes/methods from Detectron
        from modeling.model_builder import Generalized_RCNN
        from roi_data.rpn import add_rpn_blobs
        import modeling.rpn_heads as rpn_heads
        from core.test import box_results_with_nms_and_limit
        import utils.boxes as box_utils
        import utils.vis as vis_utils

        self.vis_utils = vis_utils
        self.add_rpn_blobs = add_rpn_blobs
        self.box_utils = box_utils
        self.box_results_with_nms_and_limit = box_results_with_nms_and_limit
        self.RPN = rpn_heads.generic_rpn_outputs(self.dim_out, self.spatial_scale)
        self.roi_xform = lambda x, rpn_ret: Generalized_RCNN.roi_feature_transform(
            self, x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        del sys.modules['datasets']
        sys.path.pop(0)
        self.head_batch_size = 10

    def __init__(self, basenet, args):
        super(FRCNNWrapper3, self).__init__(basenet, args)
        self.basenet = basenet
        self.init_detectron(args)
        loc_num = 2 if self.cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else args.nclass
        self.cls_loc = Unit3D(in_channels=384+384+128+128, output_channels=loc_num * 4,
                              kernel_shape=[1, 1, 1],
                              padding=0,
                              activation_fn=None,
                              use_batch_norm=False,
                              use_bias=True,
                              name='cls_loc')
        init.normal_(self.basenet.logits.conv3d.weight, std=0.01)
        init.constant_(self.basenet.logits.conv3d.bias, 0)
        init.normal_(self.cls_loc.conv3d.weight, std=0.001)
        init.constant_(self.cls_loc.conv3d.bias, 0)
        self.freeze_batchnorm = args.freeze_batchnorm
        self.n_class = args.nclass
        self.sigmoid = True
        self.input_size = args.input_size
        self.freeze_base = args.freeze_base
        self.freeze_head = args.freeze_head
        for i, end_point in enumerate(self.basenet.VALID_ENDPOINTS):
            if end_point == 'Mixed_4f':  # first half should include Mixed_4f
                self.first_layers = self.basenet.VALID_ENDPOINTS[:i+1]
                self.last_layers = self.basenet.VALID_ENDPOINTS[i+1:]

    def assign_boxes_to_labels(self, roidb, box_list):
        labels = np.zeros(len(box_list), dtype='int32')
        for i, entry in enumerate(roidb):
            ind = box_list[:, 0] == i
            boxes = box_list[ind, 1:5]
            gt_inds = np.where(entry['gt_classes'] > 0)[0]
            gt_boxes = entry['boxes'][gt_inds, :]
            gt_classes = entry['gt_classes'][gt_inds]
            proposal_to_gt_overlaps = self.box_utils.bbox_overlaps(
                boxes.astype(dtype=np.float32, copy=False),
                gt_boxes.astype(dtype=np.float32, copy=False)
            )
            argmaxes = proposal_to_gt_overlaps.argmax(axis=1)
            labels[ind] = gt_classes[argmaxes]
        return labels

    def assign_boxes_to_multilabels(self, roidb, box_list):
        labels = np.zeros((len(box_list), self.n_class), dtype='int32')
        for i, entry in enumerate(roidb):
            ind = box_list[:, 0] == i
            boxes = box_list[ind, 1:5]
            gt_inds = np.where(entry['gt_classes'] > 0)[0]
            gt_boxes = entry['boxes'][gt_inds, :]
            gt_classes = entry['gt_classes'][gt_inds]
            proposal_to_gt_overlaps = self.box_utils.bbox_overlaps(
                boxes.astype(dtype=np.float32, copy=False),
                gt_boxes.astype(dtype=np.float32, copy=False)
            )
            for j, row in enumerate(proposal_to_gt_overlaps):
                fg_inds = np.where(row > self.cfg.TRAIN.FG_THRESH)[0]
                labels[j, gt_classes[fg_inds]] = 1
            labels[labels.max(axis=1) == 0, 0] = 1  # background boxes
        return labels

    def baseforward(self, x, part):
        layers = self.first_layers if part == 'first' else self.last_layers
        for end_point in layers:
            if end_point in self.basenet.end_points:
                x = self.basenet._modules[end_point](x)
        return x

    def forward(self, im, meta):
        if self.freeze_batchnorm:
            for module in self.basenet.modules():
                if isinstance(module, torch.nn.modules.BatchNorm1d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm2d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm3d):
                    module.eval()

        # x is of the form b x n x h x w x c
        # model expects b x c x n x h x w
        x = im.permute(0, 4, 1, 2, 3)
        img_size = x.shape[3:]
        with torch.set_grad_enabled(not self.freeze_base):
            x = self.baseforward(x, 'first')

        # slice feature map to get center frame
        t = x.shape[2]
        x_slice = x[:, :, t//2, :, :]

        # pass through region proposal network
        roidb = [meta_to_entry(m, self.n_class, self.input_size) for m in meta]
        im_info = torch.ones(x_slice.shape[0], 3)
        im_info[:, :2] = self.input_size
        rpn_ret = self.RPN(x_slice, im_info, roidb=roidb)
        rpn_ret['multilabels_int32'] = self.assign_boxes_to_multilabels(roidb, rpn_ret['rois'])

        # class agnostic regression bugfix
        rpn_ret['fix_labels_int32'] = self.assign_boxes_to_labels(roidb, rpn_ret['rois'])

        print('norm rpn_cls_logits {} \t rpn_bbox_pred {}'.format(
            rpn_ret['rpn_cls_logits'].norm(),
            rpn_ret['rpn_bbox_pred'].norm())
        )
        print('mean rpn_cls_logits {} \t rpn_bbox_pred {}'.format(
            rpn_ret['rpn_cls_logits'].mean(),
            rpn_ret['rpn_bbox_pred'].mean())
        )

        # get roi features
        pools = []
        for t in range(x.shape[2]):
            x_slice = x[:, :, t, :, :]
            pools.append(self.roi_xform(x_slice, rpn_ret))
        x = torch.stack(pools, dim=2)
        del pools

        # pass through rest of the network
        with torch.set_grad_enabled(not self.freeze_head):
            x = self.baseforward(x, 'last')
            x = F.avg_pool3d(x, kernel_size=x.size()[2:])  # global avg pool

            # compute scores and locations
            x_drop = self.basenet.dropout(x)
            roi_score = self.basenet.logits(x_drop).squeeze().view(x.shape[0], -1)
        roi_cls_loc = self.cls_loc(x_drop).squeeze().view(x.shape[0], -1)

        # get bounding boxes for prediction
        rois = rpn_ret['rois']
        im_ids = rois[:, 0]
        boxes = rois[:, 1:5]
        box_deltas = roi_cls_loc.data.cpu().numpy().squeeze()
        box_deltas = box_deltas.reshape([-1, box_deltas.shape[-1]])
        if self.cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            box_deltas = box_deltas[:, -4:]
        pred_boxes = self.box_utils.bbox_transform(boxes, box_deltas, self.cfg.MODEL.BBOX_REG_WEIGHTS)
        pred_boxes = self.box_utils.clip_tiled_boxes(pred_boxes, (self.input_size, self.input_size))
        if self.cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            pred_boxes = np.tile(pred_boxes, (1, roi_score.shape[1]))
        pred_boxes = np.tile(boxes, (1, roi_score.shape[1]))  # no regression, DEBUG TODO
        if self.sigmoid:
            box_scores = F.sigmoid(roi_score.detach()).cpu().numpy()
        else:
            box_scores = F.softmax(roi_score.detach(), dim=1).cpu().numpy()
        score_predictions = []
        for i, m in enumerate(meta):
            scores, boxes, cls_boxes = self.box_results_with_nms_and_limit(box_scores[im_ids == i, :],
                                                                           pred_boxes[im_ids == i, :])
            labels = [j-1 for j in range(1, len(cls_boxes)) for _ in cls_boxes[j]]  # 0 indexed
            score_prediction = {'do_not_collate': True,
                                'vid': str2torch(m['id']).cuda(),
                                'start': torch.Tensor([(m['start'])]).cuda(),
                                'boxes': torch.Tensor(boxes).cuda() / img_size[0],
                                'labels': torch.Tensor(labels).cuda(),
                                'scores': torch.Tensor(scores).cuda()}
            score_predictions.append(score_prediction)

        # prepare data for criterion
        tensorize(rpn_ret)
        rpn_kwargs = defaultdict(list)
        rpn_kwargs['rpn_cls_logits'] = rpn_ret['rpn_cls_logits']
        rpn_kwargs['rpn_bbox_pred'] = rpn_ret['rpn_bbox_pred']
        self.add_rpn_blobs(rpn_kwargs, [1]*len(roidb), roidb)
        print('rpn pos {}\t neg {}'.format((rpn_kwargs['rpn_labels_int32_wide'] == 1).sum(),
                                           (rpn_kwargs['rpn_labels_int32_wide'] == 0).sum()))
        del rpn_kwargs['im_info'], rpn_kwargs['roidb']
        rpn_kwargs = {k: v for k, v in rpn_kwargs.items()}
        tensorize(rpn_kwargs)


        # visual debugging
        if True:
            # visualize gt
            # fake_scores = np.zeros((len(roidb[-1]['gt_classes']), 81))
            # fake_boxes = np.tile(roidb[-1]['boxes'], (1, roi_score.shape[1]))
            # for i, l in enumerate(roidb[-1]['gt_classes']):
            #     fake_scores[i, l] = 1
            # scores, boxes, cls_boxes = self.box_results_with_nms_and_limit(fake_scores, fake_boxes)

            # visualize rpn
            fake_scores = np.zeros((rpn_ret['rois'].shape[0], 81))
            fake_scores[:, -1] = np.random.rand(fake_scores.shape[0])
            fake_boxes = np.tile(rpn_ret['rois'][:, 1:5], (1, 81))
            scores, boxes, cls_boxes = self.box_results_with_nms_and_limit(fake_scores, fake_boxes)

            import types
            dataset = types.SimpleNamespace()
            setattr(dataset, 'classes', [str(cls) for cls in range(81)])
            self.vis_utils.vis_one_image(
                im[-1, im.shape[1]//2, :, :].cpu().numpy()/2 + 1/2,
                'visual',
                './',
                cls_boxes,
                thresh=.1,
                box_alpha=0.8,
                dataset=dataset,
                show_class=True
            )

        return score_predictions, roi_score, roi_cls_loc, rpn_ret, rpn_kwargs
