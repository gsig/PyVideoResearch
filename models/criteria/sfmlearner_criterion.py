from __future__ import absolute_import

from models.criteria.default_criterion import DefaultCriterion
import sys
import os.path
import os
import torch
import torch.nn.functional as F


class SfmLearnerCriterion(DefaultCriterion):
    def init_sfmlearner(self, args):
        this_dir = os.path.dirname(__file__)
        lib_path = os.path.join(this_dir, '../../external/SfmLearner-Pytorch')
        sys.path.insert(0, lib_path)

        from loss_functions import photometric_reconstruction_loss, explainability_loss, smooth_loss
        self.pr_loss = photometric_reconstruction_loss
        self.explainability_loss = explainability_loss
        self.smooth_loss = smooth_loss

    def __init__(self, args):
        super(SfmLearnerCriterion, self).__init__(args)
        self.init_sfmlearner(args)
        self.photo_loss_weight = args.photo_loss_weight
        self.mask_loss_weight = args.mask_loss_weight
        self.smooth_loss_weight = args.smooth_loss_weight
        self.inverse_loss = args.inverse_loss_weight

    def forward(self, tgt_img, ref_imgs, intrinsics, intrinsics_inv, depth, explainability_mask, pose, target, meta):
        w1, w2, w3 = self.photo_loss_weight, self.mask_loss_weight, self.smooth_loss_weight
        w4 = self.inverse_loss_weight

        loss_1 = self.pr_loss(tgt_img, ref_imgs,
                              intrinsics, intrinsics_inv,
                              depth, explainability_mask, pose)
        if w2 > 0:
            loss_2 = self.explainability_loss(explainability_mask)
        else:
            loss_2 = 0
        loss_3 = self.smooth_loss(depth)
        loss = w1*loss_1 + w2*loss_2 + w3*loss_3

        # intrinsics regularization
        b = intrinsics.size(0)
        approx_identity = torch.mm(intrinsics.view(b, 3, 3), intrinsics_inv.view(b, 3, 3))
        identity = torch.eye(3).reshape((1, 3, 3)).repeat(b, 1, 1)
        loss_4 = F.mse_loss(approx_identity, identity)
        loss += w4*loss_4

        return depth, loss
