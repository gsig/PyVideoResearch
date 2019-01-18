from models.bases.base import Base
from models.bases.resnet50_3d_encoder import ResNet3DEncoder
from models.bases.resnet50_3d_encoder import Bottleneck3D as Encoder_Bottleneck3D
from models.bases.resnet50_3d_decoder import ResNet3DDecoder
from models.bases.resnet50_3d_decoder import Bottleneck3D as Decoder_Bottleneck3D
import torch.nn as nn


class ResNet3DAutoencoder(nn.Module):
    def __init__(self, layers, num_classes=400):
        super(ResNet3DAutoencoder, self).__init__()
        self.encoder = ResNet3DEncoder(Encoder_Bottleneck3D, layers, num_classes)
        self.decoder = ResNet3DDecoder(Decoder_Bottleneck3D, layers, num_classes)

    def forward(self, x):
        code = self.encoder(x)
        x_hat = self.decoder(code)
        return x_hat, code, x


class ResNet503DAutoencoder(Base):
    @classmethod
    def get(cls, args):
        model = ResNet3DAutoencoder([3, 4, 6, 3])  # 50
        if args.pretrained:
            from torchvision.models.resnet import resnet50
            model2d = resnet50(pretrained=True)
            model.encoder.load_2d(model2d)
        return model
