"""
Temporal Relational Networks
by Bolei et al.
"""
from models.wrappers.tsn_base import TSNBase
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class RelationModuleMultiScale(torch.nn.Module):
    # Temporal Relation module in multiply scale, suming over
    # [2-frame relation, 3-frame relation, ..., n-frame relation]

    def __init__(self, img_feature_dim, num_frames, num_class):
        super(RelationModuleMultiScale, self).__init__()
        self.subsample_num = 3 # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)] # generate the multiple frame relations

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass

        self.num_class = num_class
        self.num_frames = num_frames
        num_bottleneck = 256
        self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist
        for i in range(len(self.scales)):
            scale = self.scales[i]
            fc_fusion = nn.Sequential(
                nn.ReLU(),
                nn.Linear(scale * self.img_feature_dim, num_bottleneck),
                nn.ReLU(),
                nn.Linear(num_bottleneck, self.num_class),
                )

            self.fc_fusion_scales += [fc_fusion]

        print('Multi-Scale Temporal Relation Network Module in use', ['%d-frame relation' % i for i in self.scales])

    def forward(self, input):
        # the first one is the largest scale
        act_all = input[:, self.relations_scales[0][0], :]
        act_all = act_all.view(act_all.size(0), self.scales[0] * self.img_feature_dim)
        act_all = self.fc_fusion_scales[0](act_all)

        for scaleID in range(1, len(self.scales)):
            # iterate over the scales
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]), self.subsample_scales[scaleID], replace=False)
            for idx in idx_relations_randomsample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_all += act_relation
        return act_all

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))


class TRNBase(TSNBase):
    def __init__(self, basenet, args):
        super(TRNBase, self).__init__(basenet, args)
        print('setting consensus to TRN')
        self.consensus = RelationModuleMultiScale(
            args.nclass, args.temporal_segments, args.nclass)


if __name__ == "__main__":
    batch_size = 10
    num_frames = 5
    num_class = 174
    img_feature_dim = 512
    input_var = Variable(torch.randn(batch_size, num_frames, img_feature_dim))
    model = RelationModuleMultiScale(img_feature_dim, num_frames, num_class)
    output = model(input_var)
    print(output)
