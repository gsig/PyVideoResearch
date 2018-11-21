from models.bases.base import Base
import torch.nn as nn
from collections import OrderedDict


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck3D, self).__init__()
        #self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1, 1, 1),
        #                       padding=(0, 0, 0), bias=False)
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1),
                               padding=(1, 0, 0), bias=False)
        #self.bn1 = nn.BatchNorm3d(planes)
        self.bn1 = nn.BatchNorm3d(planes, eps=0.001, momentum=0.01)
        #self.conv2 = nn.Conv3d(planes, planes, kernel_size=(3, 3, 3),
        #                       stride=(1, stride, stride), padding=(1, 1, 1), bias=False)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3),
                               stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        #self.bn2 = nn.BatchNorm3d(planes)
        self.bn2 = nn.BatchNorm3d(planes, eps=0.001, momentum=0.01)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion,
                               kernel_size=(1, 1, 1), bias=False)
        #self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):

    def __init__(self, block, layers, num_classes=400):
        super(ResNet3D, self).__init__()
        self.global_pooling = True
        self.inplanes = 64
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=2,
                               padding=(2, 3, 3), bias=False)
        #self.bn1 = nn.BatchNorm3d(64)
        self.bn1 = nn.BatchNorm3d(64, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3,
                                    stride=2, padding=(1, 1, 1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.maxpool2 = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=(1, 1, 1))
        self.dropout = nn.Dropout(.5)
        self.fc = nn.Conv3d(512 * block.expansion, num_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1, stride, stride),
                          bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x is of the form b x n x h x w x c
        # model expects b x c x n x h x w
        x = x.permute(0, 4, 1, 2, 3)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        logits = self.fc(x)

        logits = logits.mean(3).mean(3)
        # model returns batch x classes x time
        logits = logits.permute(0, 2, 1)
        # logits is batch X time X classes
        if self.global_pooling:
            logits = logits.mean(1)
        return logits

    def load_2d(self, model2d):
        print('inflating 2d resnet parameters')
        sd = self.state_dict()
        sd2d = model2d.state_dict()
        sd = OrderedDict([(x.replace('module.', ''), y) for x, y in sd.items()])
        sd2d = OrderedDict([(x.replace('module.', ''), y) for x, y in sd2d.items()])
        for k, v in sd2d.items():
            if k not in sd:
                print('ignoring state key for loading: {}'.format(k))
                continue
            if 'conv' in k or 'downsample.0' in k:
                s = sd[k].shape
                t = s[2]
                sd[k].copy_(v.unsqueeze(2).expand(*s) / t)
            elif 'bn' in k or 'downsample.1' in k:
                sd[k].copy_(v)
            else:
                print('skipping: {}'.format(k))

    def replace_logits(self, num_classes):
        self.fc = nn.Conv3d(self.fc.in_channels, num_classes, kernel_size=1)


class ResNet503D(Base):
    @classmethod
    def get(cls, args):
        model = ResNet3D(Bottleneck3D, [3, 4, 6, 3])  # 50
        if args.pretrained:
            from torchvision.models.resnet import resnet50
            model2d = resnet50(pretrained=True)
            model.load_2d(model2d)
        return model


if __name__ == "__main__":
    import torch
    batch_size = 8
    num_frames = 32
    img_feature_dim = 224
    input_var = torch.randn(batch_size, num_frames, img_feature_dim, img_feature_dim, 3).cuda()
    model = ResNet503D.get(None)
    model = model.cuda()
    output = model(input_var)
    print(output.shape)
