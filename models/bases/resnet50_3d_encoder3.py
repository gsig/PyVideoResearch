from models.bases.base import Base
from models.bases.resnet50_3d import ResNet3D, Bottleneck3D


class ResNet3DEncoder(ResNet3D):
    def __init__(self, block, layers, num_classes=400):
        super(ResNet3DEncoder, self).__init__(block, layers, num_classes)
        self.fc = None

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
        #x = self.layer3(x)
        #x = self.layer4(x)

        #x = self.avgpool(x)
        #x = self.dropout(x)
        #logits = self.fc(x)

        #logits = logits.mean(3).mean(3)
        # model returns batch x classes x time
        #logits = logits.permute(0, 2, 1)
        # logits is batch X time X classes
        #if self.global_pooling:
        #    logits = logits.mean(1)
        #return logits
        return x

class ResNet503DEncoder(Base):
    @classmethod
    def get(cls, args):
        model = ResNet3DEncoder(Bottleneck3D, [3, 4, 6, 3])  # 50
        if args.pretrained:
            from torchvision.models.resnet import resnet50
            model2d = resnet50(pretrained=True)
            model.load_2d(model2d)
        return model
