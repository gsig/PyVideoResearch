import unittest
from models.criteria import default_criterion
import torch
import torch.optim
import torch.testing


class Args():
    pass


def test_model_updates(inputs, model, target):
    args = Args()
    args.balanceloss = False
    args.window_smooth = 0
    criterion = default_criterion.DefaultCriterion(args)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    optimizer.zero_grad()
    params = [np for np in model.named_parameters()]
    initial_params = [(name, p.clone()) for (name, p) in params]
    output = model(inputs)
    meta = {}
    _, loss, _ = criterion(output, target, meta)
    loss.backward()
    optimizer.step()
    for (_, p0), (name, p1) in zip(initial_params, params):
        try:
            assert not torch.equal(p0, p1)
        except AssertionError:
            print(name)
            import pdb
            pdb.set_trace()
            raise


class TestBases(unittest.TestCase):
    def test_aj_i3d_updates(self):
        torch.manual_seed(12345)
        from models.bases import aj_i3d
        b, f, d = 2, 32, 224
        inputs = torch.randn(b, f, d, d, 3)
        model = aj_i3d.AJ_I3D.get(None)
        target = torch.zeros(2, 32, 157)
        test_model_updates(inputs, model, target)

    def test_resnet50_3d_updates(self):
        torch.manual_seed(12345)
        from models.bases import resnet50_3d
        b, f, d = 2, 32, 224
        inputs = torch.randn(b, f, d, d, 3)
        args = Args()
        args.pretrained = False
        model = resnet50_3d.ResNet503D.get(args)
        target = torch.zeros(2, 400)
        test_model_updates(inputs, model, target)

    def test_resnet50_3d_nonlocal_updates(self):
        torch.manual_seed(12345)
        from models.bases import resnet50_3d_nonlocal
        b, f, d = 2, 32, 224
        inputs = torch.randn(b, f, d, d, 3)
        args = Args()
        args.pretrained = False
        model = resnet50_3d_nonlocal.ResNet503DNonLocal.get(args)
        target = torch.zeros(2, 400)
        test_model_updates(inputs, model, target)
