"""
"""
from models.bases.resnet50_3d import ResNet3D
from collections import OrderedDict


class FeatureExtractorWrapper(ResNet3D):
    def __init__(self, basenet, opts, *args, **kwargs):
        if 'DataParallel' in basenet.__class__.__name__:
            basenet = basenet.module
        print('Initializing feature extractor with ResNet instance')
        self.__dict__ = basenet.__dict__
        self.output_features = opts.features.split(';')

    def forward(self, x):
        output = OrderedDict()
        # x is of the form b x n x h x w x c
        # model expects b x c x n x h x w
        x = x.permute(0, 4, 1, 2, 3)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if 'conv1' in self.output_features:
            output['conv1'] = x

        x = self.layer1(x)
        x = self.maxpool2(x)
        if 'layer1' in self.output_features:
            output['layer1'] = x
        x = self.layer2(x)
        if 'layer2' in self.output_features:
            output['layer2'] = x
        x = self.layer3(x)
        if 'layer3' in self.output_features:
            output['layer3'] = x
        x = self.layer4(x)
        if 'layer4' in self.output_features:
            output['layer4'] = x

        x = self.avgpool(x)
        x = self.dropout(x)
        logits = self.fc(x)

        logits = logits.mean(3).mean(3)
        # model returns batch x classes x time
        logits = logits.permute(0, 2, 1)
        # logits is batch X time X classes
        if self.global_pooling:
            logits = logits.mean(1)
        if 'fc' in self.output_features:
            output['fc'] = x
        return output
