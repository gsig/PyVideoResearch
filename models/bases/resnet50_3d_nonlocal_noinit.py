from models.bases.base import Base
from models.bases.resnet50_3d import Bottleneck3D, ResNet3D
from models.layers.nonlocal_block_3d_no_init import NONLocalBlock3DNoInit as NONLocalBlock3D
import torch.nn as nn


class ResNet3DNonLocal(ResNet3D):
    def __init__(self, block, layers, num_classes=400):
        super(ResNet3DNonLocal, self).__init__(block, layers, num_classes)

    def insert_nonlocal_blocks(self, nonlocal_blocks):
        for layername, nr in zip(['layer1', 'layer2', 'layer3', 'layer4'], nonlocal_blocks):
            if nr == 0:
                continue
            layers = getattr(self, layername)
            newlayers = []
            insert_freq = len(layers) / nr
            for i, layer in enumerate(layers):
                newlayers.append(layer)
                if i % insert_freq == 0:
                    n = layer.conv3.out_channels
                    if layername == 'layer2':
                        blocknl = NONLocalBlock3D(n, group_size=4)
                    else:
                        blocknl = NONLocalBlock3D(n)
                    newlayers.append(blocknl)

            newlayers = nn.Sequential(*newlayers)
            setattr(self, layername, newlayers)


class ResNet503DNonLocalNoInit(Base):
    @classmethod
    def get(cls, args):
        model = ResNet3DNonLocal(Bottleneck3D, [3, 4, 6, 3])  # 50
        if args.pretrained:
            from torchvision.models.resnet import resnet50
            model2d = resnet50(pretrained=True)
            model.load_2d(model2d)
        model.insert_nonlocal_blocks([0, 2, 3, 0])
        return model


if __name__ == "__main__":
    import torch
    batch_size = 8
    num_frames = 32
    img_feature_dim = 224
    input_var = torch.randn(batch_size, num_frames, img_feature_dim, img_feature_dim, 3).cuda()
    args = {}
    setattr(args, 'pretrained', True)
    model = ResNet503DNonLocalNoInit.get(args)
    model = model.cuda()
    output = model(input_var)
    print(output.shape)
