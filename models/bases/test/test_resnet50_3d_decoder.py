import unittest
import torch
import torch.nn as nn


class Args():
    pass


def test_model_updates(inputs, model, target, criterion, meta={}):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    optimizer.zero_grad()
    params = [np for np in model.named_parameters()]
    initial_params = [(name, p.clone()) for (name, p) in params]
    output = model(inputs, meta)
    #if type(output) is not tuple:
    #    output = (output, )
    #_, loss, _ = criterion(*(output + (target, meta)))
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    for (_, p0), (name, p1) in zip(initial_params, params):
        try:
            assert not torch.equal(p0, p1)
        except AssertionError:
            print(name)
            raise


class TestBases(unittest.TestCase):
    def test_resnet50_3d_decoder_updates(self):
        torch.manual_seed(12345)
        from models.bases.resnet50_3d_encoder import ResNet503DEncoder
        from models.bases.resnet50_3d_decoder import ResNet503DDecoder
        from models.wrappers.default_wrapper import DefaultWrapper
        args = Args()
        args.pretrained = False
        args.freeze_batchnorm = False
        encoder = ResNet503DEncoder.get(args)
        decoder = ResNet503DDecoder.get(args)
        model = nn.Sequential(encoder, decoder)
        model = DefaultWrapper(model, args)
        criterion = nn.MSELoss()
        b, f, d = 2, 32, 224
        inputs = torch.randn(b, f, d, d, 3)
        target = inputs
        test_model_updates(inputs, model, target, criterion)
