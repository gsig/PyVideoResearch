from __future__ import absolute_import

from models.wrappers.wrapper import Wrapper
import sys
import os.path
import os
import torch.nn as nn
import torch


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


def modify_intrinsics(intrinsics, intrinsics_type):
    if intrinsics_type == 'full':
        pass
    elif intrinsics_type == 'linear':
        intrinsics[:, 0, 0] = torch.exp(intrinsics[:, 0, 0])
        intrinsics[:, 1, 1] = torch.exp(intrinsics[:, 1, 1])
        intrinsics[:, 2, 2] = torch.exp(intrinsics[:, 2, 2])
        intrinsics[:, 0, 1] = 0
        intrinsics[:, 1, 0] = 0
        intrinsics[:, 2, 0] = 0
        intrinsics[:, 2, 1] = 0
    elif intrinsics_type == 'scaled':
        intrinsics[:, 0, 0] = 7.215377e+02 * torch.exp(intrinsics[:, 0, 0])
        intrinsics[:, 1, 1] = 7.215377e+02 * torch.exp(intrinsics[:, 1, 1])
        intrinsics[:, 2, 2] = 1.000000e+00 * torch.exp(intrinsics[:, 2, 2])
        intrinsics[:, 0, 2] = 6.095593e+02 * torch.exp(intrinsics[:, 0, 2])
        intrinsics[:, 1, 2] = 1.728540e+02 * torch.exp(intrinsics[:, 1, 2])
        intrinsics[:, 0, 1] = 0
        intrinsics[:, 1, 0] = 0
        intrinsics[:, 2, 0] = 0
        intrinsics[:, 2, 1] = 0
    elif intrinsics_type == 'fixed':
        # KITTI camera 3
        # 3x4 projection matrix after rectification
        # 7.215377e+02 0.000000e+00 6.095593e+02 -3.395242e+02
        # 0.000000e+00 7.215377e+02 1.728540e+02 2.199936e+00
        # 0.000000e+00 0.000000e+00 1.000000e+00 2.729905e-03
        #
        # 3x3 inverted:
        # array([[ 0.00138593,  0.        , -0.84480589],
        #        [ 0.        ,  0.00138593, -0.23956337],
        #        [ 0.        ,  0.        ,  1.        ]])
        intrinsics[:, 0, 0] = 7.215377e+02
        intrinsics[:, 1, 1] = 7.215377e+02
        intrinsics[:, 2, 2] = 1.000000e+00
        intrinsics[:, 0, 2] = 6.095593e+02
        intrinsics[:, 1, 2] = 1.728540e+02
        intrinsics[:, 0, 1] = 0
        intrinsics[:, 1, 0] = 0
        intrinsics[:, 2, 0] = 0
        intrinsics[:, 2, 1] = 0
    else:
        assert False, "wrong intrinsics-type"
    return intrinsics


def modify_intrinsics_inv(intrinsics, intrinsics_type):
    if intrinsics_type == 'full':
        pass
    elif intrinsics_type == 'linear':
        intrinsics[:, 0, 0] = torch.exp(intrinsics[:, 0, 0])
        intrinsics[:, 1, 1] = torch.exp(intrinsics[:, 1, 1])
        intrinsics[:, 2, 2] = torch.exp(intrinsics[:, 2, 2])
        intrinsics[:, 0, 1] = 0
        intrinsics[:, 1, 0] = 0
        intrinsics[:, 2, 0] = 0
        intrinsics[:, 2, 1] = 0
    elif intrinsics_type == 'scaled':
        intrinsics[:, 0, 0] = 0.00138593 * torch.exp(intrinsics[:, 0, 0])
        intrinsics[:, 1, 1] = 0.00138593 * torch.exp(intrinsics[:, 1, 1])
        intrinsics[:, 2, 2] = 1 * torch.exp(intrinsics[:, 2, 2])
        intrinsics[:, 0, 2] = -0.84480589 * torch.exp(intrinsics[:, 0, 2])
        intrinsics[:, 1, 2] = -0.23956337 * torch.exp(intrinsics[:, 1, 2])
        intrinsics[:, 0, 1] = 0
        intrinsics[:, 1, 0] = 0
        intrinsics[:, 2, 0] = 0
        intrinsics[:, 2, 1] = 0
    elif intrinsics_type == 'fixed':
        # KITTI camera 3
        # 3x4 projection matrix after rectification
        # 7.215377e+02 0.000000e+00 6.095593e+02 -3.395242e+02
        # 0.000000e+00 7.215377e+02 1.728540e+02 2.199936e+00
        # 0.000000e+00 0.000000e+00 1.000000e+00 2.729905e-03
        #
        # 3x3 inverted:
        # array([[ 0.00138593,  0.        , -0.84480589],
        #        [ 0.        ,  0.00138593, -0.23956337],
        #        [ 0.        ,  0.        ,  1.        ]])
        intrinsics[:, 0, 0] = 0.00138593
        intrinsics[:, 1, 1] = 0.00138593
        intrinsics[:, 2, 2] = 1
        intrinsics[:, 0, 2] = -0.84480589
        intrinsics[:, 1, 2] = -0.23956337
        intrinsics[:, 0, 1] = 0
        intrinsics[:, 1, 0] = 0
        intrinsics[:, 2, 0] = 0
        intrinsics[:, 2, 1] = 0
    else:
        assert False, "wrong intrinsics-type"
    return intrinsics


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
                intrinsics = modify_intrinsics(intrinsics, args.intrinsics_type)

                out_conv6c = self.conv6b(conv5)
                out_conv7c = self.conv7b(out_conv6c)
                intrinsics_inv = self.intrinsics_inv_pred(out_conv7c)
                intrinsics_inv = intrinsics_inv.mean(3).mean(2)
                intrinsics_inv = 0.01 * intrinsics_inv.view(intrinsics_inv.size(0), self.nb_ref_imgs, 9)
                intrinsics_inv = intrinsics_inv.mean(1).view(intrinsics_inv.size(0), 3, 3)
                intrinsics_inv = modify_intrinsics_inv(intrinsics_inv, args.intrinsics_type)

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
        if type(disparities) == tuple:
            depth = [1/disp for disp in disparities]
        else:
            depth = 1/disparities
        explainability_mask, pose, intrinsics, intrinsics_inv = self.pose_exp_net(tgt_img, ref_imgs)

        # debug
        print('intrinsics:')
        print(intrinsics.mean(0).view(-1))
        print('intrinsics_inv:')
        print(intrinsics_inv.mean(0).view(-1))

        return tgt_img, ref_imgs, intrinsics, intrinsics_inv, depth, explainability_mask, pose
