from __future__ import absolute_import

from models.wrappers.wrapper import Wrapper
import sys
import os.path
import os
import torch.nn as nn


class HookModule(nn.Module):
    def __init__(self, module):
        super(HookModule, self).__init__()
        self.module = module
        self.storage = None

    def forward(self, x):
        self.storage = self.module(x)
        return self.storage

    def purge(self):
        self.storage = None


class SfmLearnerWrapper(Wrapper):
    def init_sfmlearner(self, args):
        if 'models' in sys.modules:
            del sys.modules['models']
        this_dir = os.path.dirname(__file__)
        lib_path = os.path.join(this_dir, '../../external/SfmLearner-Pytorch')
        sys.path.insert(0, lib_path)

        from models.DispNetS import DispNetS
        from models.PoseExpNet import PoseExpNet, conv
        self.disp_net = DispNetS()

        class IntrinsicsPoseExpNet(PoseExpNet, object):
            def __init__(self, nb_ref_imgs=2, output_exp=False):
                super(IntrinsicsPoseExpNet, self).__init__(nb_ref_imgs, output_exp)
                self.conv5 = HookModule(self.conv5)
                conv_planes = [16, 32, 64, 128, 256, 256, 256]
                self.conv6b = conv(conv_planes[4], conv_planes[5])
                self.conv7b = conv(conv_planes[5], conv_planes[6])
                self.intrinsics_pred = nn.Conv2d(conv_planes[6], 9*self.nb_ref_imgs, kernel_size=1, padding=0)

                self.conv6c = conv(conv_planes[4], conv_planes[5])
                self.conv7c = conv(conv_planes[5], conv_planes[6])
                self.intrinsics_inv_pred = nn.Conv2d(conv_planes[6], 9*self.nb_ref_imgs, kernel_size=1, padding=0)

            def forward(self, target_image, ref_imgs):
                exp_mask, pose = super(IntrinsicsPoseExpNet, self).forward(target_image, ref_imgs)
                conv5 = self.conv5.storage
                out_conv6b = self.conv6b(conv5)
                out_conv7b = self.conv7b(out_conv6b)
                intrinsics = self.intrinsics_pred(out_conv7b)
                intrinsics = intrinsics.mean(3).mean(2)
                intrinsics = 0.01 * intrinsics.view(intrinsics.size(0), self.nb_ref_imgs, 9)
                intrinsics = intrinsics.mean(1).view(intrinsics.size(0), 3, 3)

                out_conv6c = self.conv6b(conv5)
                out_conv7c = self.conv7b(out_conv6c)
                intrinsics_inv = self.intrinsics_inv_pred(out_conv7c)
                intrinsics_inv = intrinsics_inv.mean(3).mean(2)
                intrinsics_inv = 0.01 * intrinsics_inv.view(intrinsics_inv.size(0), self.nb_ref_imgs, 9)
                intrinsics_inv = intrinsics_inv.mean(1).view(intrinsics_inv.size(0), 3, 3)

                self.conv5.purge()
                return exp_mask, pose, intrinsics, intrinsics_inv
        self.pose_exp_net = IntrinsicsPoseExpNet(nb_ref_imgs=1, output_exp=True)
        del sys.modules['models']
        sys.path.pop(0)

    def __init__(self, basenet, args):
        super(SfmLearnerWrapper, self).__init__(basenet, args)
        self.init_sfmlearner(args)

    def forward(self, im, meta):
        # im is of the form b x n x h x w x c
        im = im.permute(0, 4, 1, 2, 3)
        tgt_img = im[:, :, 0, :, :]
        ref_imgs = [im[:, :, i, :, :] for i in range(1, im.size(2))]

        # compute output
        disparities = self.disp_net(tgt_img)
        depth = [1/disp for disp in disparities]
        explainability_mask, pose, intrinsics, intrinsics_inv = self.pose_exp_net(tgt_img, ref_imgs)

        return tgt_img, ref_imgs, intrinsics, intrinsics_inv, depth, explainability_mask, pose
