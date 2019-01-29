from __future__ import absolute_import

from models.wrappers.wrapper import Wrapper
import sys
import os.path
import os
import torch.nn as nn


class SfmLearnerWrapper(Wrapper):
    def init_sfmlearner(self, args):
        def init_sfmlearner(self, args):
            if 'models' in sys.modules:
                del sys.modules['models']
            this_dir = os.path.dirname(__file__)
            lib_path = os.path.join(this_dir, '../../external/SfmLearner-Pytorch')
            sys.path.insert(0, lib_path)

            from models import DispNetS, PoseExpNet
            from models.PoseExpNet import conv
            self.disp_net = DispNetS()

            class IntrinsicsPoseExpNet(PoseExpNet, object):
                def __init__(self, nb_ref_imgs=2, output_exp=False):
                    super(IntrinsicsPoseExpNet, self).__init__(nb_ref_imgs, output_exp)
                    self.conv5_output = None
                    self.old_conv5 = self.conv5

                    def conv5_fun(x):
                        self.conv5_output = self.old_conv5(x)
                        return self.conv5_output
                    self.conv5 = conv5_fun
                    conv_planes = [16, 32, 64, 128, 256, 256, 256]
                    self.conv6b = conv(conv_planes[4], conv_planes[5])
                    self.conv7b = conv(conv_planes[5], conv_planes[6])
                    self.intrinsics_pred = nn.Conv2d(conv_planes[6], 9*self.nb_ref_imgs, kernel_size=1, padding=0)

                    self.conv6c = conv(conv_planes[4], conv_planes[5])
                    self.conv7c = conv(conv_planes[5], conv_planes[6])
                    self.intrinsics_inv_pred = nn.Conv2d(conv_planes[6], 9*self.nb_ref_imgs, kernel_size=1, padding=0)

                def forward(self, target_image, ref_imgs):
                    exp_mask, pose = super(IntrinsicsPoseExpNet, self).forward(target_image, ref_imgs)
                    conv5 = self.conv5_output
                    out_conv6b = self.conv6b(conv5)
                    out_conv7b = self.conv7b(out_conv6b)
                    intrinsics = self.intrinsics_pred(out_conv7b)
                    intrinsics = intrinsics.mean(3).mean(2)
                    intrinsics = 0.01 * intrinsics.view(intrinsics.size(0), self.nb_ref_imgs, 9)

                    out_conv6c = self.conv6b(conv5)
                    out_conv7c = self.conv7b(out_conv6c)
                    intrinsics_inv = self.intrinsics_inv_pred(out_conv7c)
                    intrinsics_inv = intrinsics_inv.mean(3).mean(2)
                    intrinsics_inv = 0.01 * intrinsics_inv.view(intrinsics_inv.size(0), self.nb_ref_imgs, 9)

                    self.conv5_output = None
                    return exp_mask, pose, intrinsics, intrinsics_inv
            self.pose_exp_net = IntrinsicsPoseExpNet(nb_ref_imgs=1, output_exp=True)
            del sys.modules['models']
            sys.path.pop(0)

    def __init__(self, basenet, args):
        super(SfmLearnerWrapper, self).__init__(basenet, args)
        self.init_sfmlearner(args)

    def forward(self, im, meta):
        # im is of the form b x n x h x w x c
        tgt_img = im[:, 0, :, :, :]
        ref_imgs = [im[:, i, :, :, :] for i in range(1, im.size(1))]

        # compute output
        disparities = self.disp_net(tgt_img)
        depth = [1/disp for disp in disparities]
        explainability_mask, pose, intrinsics, intrinsics_inv = self.pose_exp_net(tgt_img, ref_imgs)

        return tgt_img, ref_imgs, intrinsics, intrinsics_inv, depth, explainability_mask, pose
