# @chenyuntc

from wrapper import Wrapper
import torch
import sys
import os.path
from torch.nn import functional as F
from models.bases.aj_i3d import Unit3D
import numpy as np
import cupy as cp


class FRCNNWrapper(Wrapper):
    def __init__(self, basenet, args):
        super(FRCNNWrapper, self).__init__(basenet, args)
        self.basenet = basenet
        #del sys.modules['utils']
        this_dir = os.path.dirname(__file__)
        lib_path = os.path.join(this_dir, 'simple-faster-rcnn-pytorch')
        sys.path.insert(0, lib_path)
        from utils.array_tool import totensor, tonumpy
        from model.region_proposal_network import RegionProposalNetwork
        from model.roi_module import RoIPooling2D
        from model.utils.nms import non_maximum_suppression
        from model.utils.bbox_tools import loc2bbox
        from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator
        self.totensor = totensor
        self.tonumpy = tonumpy
        self.proposal_target_creator = ProposalTargetCreator()
        self.anchor_target_creator = AnchorTargetCreator()
        self.nms = non_maximum_suppression
        self.loc2bbox = loc2bbox
        ratios = [0.5, 1, 2]
        anchor_scales = [8, 16, 32]
        feat_stride = 16
        self.rpn = RegionProposalNetwork(
            528, 528,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=feat_stride,
        )
        roi_size = 7
        spatial_scale = 1. / feat_stride
        self.roi = RoIPooling2D(roi_size, roi_size, spatial_scale)
        self.cls_loc = Unit3D(in_channels=384+384+128+128, output_channels=args.nclass * 4,
                              kernel_shape=[1, 1, 1],
                              padding=0,
                              activation_fn=None,
                              use_batch_norm=False,
                              use_bias=True,
                              name='cls_loc')
        self.loc_normalize_mean = (0., 0., 0., 0.)
        self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
        self.nms_thresh = 0.3
        self.score_thresh = 0  # 0.01  # 0.05
        self.n_class = args.nclass
        for i, end_point in enumerate(self.basenet.VALID_ENDPOINTS):
            if end_point == 'Mixed_4f':
                self.first_layers = self.basenet.VALID_ENDPOINTS[:i]
                self.last_layers = self.basenet.VALID_ENDPOINTS[i:]
        self.head_batch_size = 30
        sys.path.pop(0)

    def baseforward(self, x, part):
        layers = self.first_layers if part == 'first' else self.last_layers
        for end_point in layers:
            if end_point in self.basenet.end_points:
                x = self.basenet._modules[end_point](x)
        return x

    def forward(self, x, meta):
        #for module in self.basenet.modules():
        #    if isinstance(module, torch.nn.modules.BatchNorm3d):
        #        module.eval()
        # x is of the form b x n x h x w x c
        # model expects b x c x n x h x w
        x = x.permute(0, 4, 1, 2, 3)
        img_size = x.shape[3:]
        x = self.baseforward(x, 'first')

        # slice feature map to get center frame
        t = x.shape[2]
        x_slice = x[:, :, t/2, :, :]  # .permute(0, 3, 1, 2)
        scale = 1

        # pass through region proposal network
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(x_slice, img_size, scale)

        bbox = meta['box']
        #label = torch.Tensor(meta['labels'])
        label = meta['label']
        if bbox.dim() == 3:
            bbox = bbox[0]
        if type(bbox) == list and len(bbox) > 0:
            bbox = torch.stack(bbox).t()  # collate transposes the tensor
        bbox = bbox * img_size[0]
        bbox = bbox[:, [1, 0, 3, 2]]  # switch to y1, x1, y2, x2
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            rois,
            self.tonumpy(bbox),
            self.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        sample_roi_index = torch.zeros(len(sample_roi))

        # TODO DEBUG
        if True or self.training:
            rois = sample_roi
            roi_indices = sample_roi_index
        else:
            gt_roi_label = torch.zeros(len(rois))
            gt_roi_loc = torch.zeros(len(rois), 4)

        # extract feature map for each proposal
        roi_indices = self.totensor(roi_indices).float()
        rois = self.totensor(rois).float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()
        pools = []
        for t in range(x.shape[2]):
            x_slice = x[:, :, t, :, :]
            pool = self.roi(x_slice, indices_and_rois)
            pools.append(pool)
        x = torch.stack(pools, dim=2)

        output_chunks = []
        for chunk in x.split(self.head_batch_size):
            output_chunks.append(self.baseforward(chunk, 'last'))
        x = torch.cat(output_chunks)

        x = x.mean(2, keepdim=True).mean(3, keepdim=True).mean(4, keepdim=True)
        roi_score = self.basenet.logits(self.basenet.dropout(x)).squeeze()
        roi_cls_loc = self.cls_loc(self.basenet.dropout(x)).squeeze()

        bbox, label, score = self.convert_prediction_to_bbox(roi_cls_loc, rois, roi_score, img_size)
        ind = score.argsort()[-300:]
        bbox = bbox[ind, :]
        label = label[ind]
        score = score[ind]
        score_prediction = {'boxes': bbox[:, [0, 1, 3, 2]] / img_size[0],  # switch to x1, y1, x2, y2
                            'labels': label,
                            'scores': score}

        return score_prediction, roi_score, roi_cls_loc, rpn_scores, rpn_locs, anchor, gt_roi_loc, gt_roi_label

    def convert_prediction_to_bbox(self, roi_cls_loc, roi, roi_score, size):
        # Convert predictions to bounding boxes in image coordinates.
        # Bounding boxes are scaled to the scale of the input images.
        mean = torch.Tensor(self.loc_normalize_mean).cuda(). \
            repeat(self.n_class)[None]
        std = torch.Tensor(self.loc_normalize_std).cuda(). \
            repeat(self.n_class)[None]

        roi_cls_loc = (roi_cls_loc * std + mean)
        roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
        roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
        cls_bbox = self.loc2bbox(self.tonumpy(roi).reshape((-1, 4)),
                                 self.tonumpy(roi_cls_loc).reshape((-1, 4)))
        cls_bbox = self.totensor(cls_bbox)
        cls_bbox = cls_bbox.view(-1, self.n_class * 4)
        # clip bounding box
        cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
        cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

        #prob = self.tonumpy(F.softmax(self.totensor(roi_score), dim=1))
        prob = self.tonumpy(F.sigmoid(self.totensor(roi_score)))

        raw_cls_bbox = self.tonumpy(cls_bbox)
        raw_prob = self.tonumpy(prob)

        bbox, label, score = self._suppress(raw_cls_bbox, raw_prob)
        return bbox, label, score

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = self.nms(cp.array(cls_bbox_l), self.nms_thresh, prob_l)
            keep = cp.asnumpy(keep)
            bbox.append(cls_bbox_l[keep])
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score
