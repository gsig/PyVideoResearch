from base import Base
from resnet50_3d import Bottleneck3D, ResNet3D


class ResNet1013D(Base):
    @classmethod
    def get(cls, args):
        model = ResNet3D(Bottleneck3D, [3, 8, 36, 3])  # 101
        if args.pretrained:
            from torchvision.models.resnet import resnet101
            model2d = resnet101(pretrained=True)
            model.load_2d(model2d)
        return model


if __name__ == "__main__":
    import torch
    batch_size = 8
    num_frames = 32
    img_feature_dim = 224
    input_var = torch.randn(batch_size, num_frames, img_feature_dim, img_feature_dim, 3).cuda()
    model = ResNet1013D.get(None)
    model = model.cuda()
    output = model(input_var)
    print(output.shape)
