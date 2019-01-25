from models.bases.base import Base
import torch.nn as nn
from collections import OrderedDict
import torch


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck3D, self).__init__()
        self.conv1 = nn.ConvTranspose3d(planes, inplanes, kernel_size=(3, 1, 1),
                                        padding=(1, 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(inplanes, eps=0.001, momentum=0.01)
        output_padding = (0, 1, 1) if stride == 2 else 0
        self.conv2 = nn.ConvTranspose3d(planes, planes, kernel_size=(1, 3, 3), output_padding=output_padding,
                                        stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes, eps=0.001, momentum=0.01)
        self.conv3 = nn.ConvTranspose3d(planes * self.expansion, planes,
                                        kernel_size=(1, 1, 1), bias=False)
        self.bn3 = nn.BatchNorm3d(planes, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv3(x)
        out = self.bn3(out)

        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)

        #out = self.conv1(x)
        #out = self.bn1(out)
        #out = self.relu(out)

        #out = self.conv2(out)
        #out = self.bn2(out)
        #out = self.relu(out)

        #out = self.conv3(out)
        #out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet3DDecoder(nn.Module):

    def __init__(self, block, layers, num_classes=400):
        super(ResNet3DDecoder, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.ConvTranspose3d(64, 3, output_padding=1,
                                        kernel_size=(5, 7, 7), stride=2, padding=(2, 3, 3),
                                        bias=False)
        self.bn1 = nn.BatchNorm3d(64, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.ConvTranspose3d(self.inplanes, self.inplanes, output_padding=1,
                                          kernel_size=3, stride=2, padding=(1, 1, 1),
                                          bias=False)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.maxpool2 = nn.ConvTranspose3d(self.inplanes, self.inplanes, output_padding=(1, 0, 0),
                                           kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0),
                                           bias=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            output_padding = (0, 1, 1) if stride == 2 else 0
            downsample = nn.Sequential(
                nn.ConvTranspose3d(planes * block.expansion, self.inplanes, output_padding=output_padding,
                                   kernel_size=1, stride=(1, stride, stride),
                                   bias=False),
                nn.BatchNorm3d(self.inplanes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        # return nn.Sequential(*layers)
        return nn.Sequential(*layers[::-1])

    def forward(self, x):
        # x is 2, 2048, 4, 7, 7
        # (2, 2048, 4, 7, 7)
        #x = self.layer4(x)
        # (2, 1024, 4, 14, 14)
        #x = self.layer3(x)
        # (2, 512, 4, 28, 28)
        x = self.layer2(x)
        x = self.maxpool2(x)
        # (2, 256, 4, 56, 56)
        x = self.layer1(x)
        # (2, 64, 7, 56, 56)

        x = self.maxpool(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.tanh(x)
        # model expects b x c x n x h x w
        # x is of the form b x n x h x w x c
        x = x.permute(0, 2, 3, 4, 1)

        # match the range of the normalized data
        x = x * .5 + .5
        x = x - torch.Tensor([0.485, 0.456, 0.406])[None, None, None, None, :].to(x.device)
        x = x / torch.Tensor([0.229, 0.224, 0.225])[None, None, None, None, :].to(x.device)
        return x

#    def load_2d(self, model2d):
#        print('inflating 2d resnet parameters')
#        sd = self.state_dict()
#        sd2d = model2d.state_dict()
#        sd = OrderedDict([(x.replace('module.', ''), y) for x, y in sd.items()])
#        sd2d = OrderedDict([(x.replace('module.', ''), y) for x, y in sd2d.items()])
#        for k, v in sd2d.items():
#            if k not in sd:
#                print('ignoring state key for loading: {}'.format(k))
#                continue
#            if 'conv' in k or 'downsample.0' in k:
#                s = sd[k].shape
#                t = s[2]
#                sd[k].copy_(v.unsqueeze(2).expand(*s) / t)
#            elif 'bn' in k or 'downsample.1' in k:
#                sd[k].copy_(v)
#            else:
#                print('skipping: {}'.format(k))


class ResNet503DDecoder3(Base):
    @classmethod
    def get(cls, args):
        model = ResNet3DDecoder(Bottleneck3D, [3, 4, 6, 3])  # 50
        #if args.pretrained:
        #    from torchvision.models.resnet import resnet50
        #    model2d = resnet50(pretrained=True)
        #    model.load_2d(model2d)
        return model
