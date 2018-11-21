import unittest
from models.criteria import default_criterion
import torch
import torch.optim
import numpy as np
import torch.testing


class Args():
    pass


def test_model_updates(inputs, model, target, whitelist=[]):
    args = Args()
    args.balanceloss = False
    args.window_smooth = 0
    criterion = default_criterion.DefaultCriterion(args)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.)
    optimizer.zero_grad()
    params = list(model.named_parameters())
    initial_params = [(name, p.clone()) for (name, p) in params]
    output = model(inputs)
    meta = {}
    _, loss, _ = criterion(output, target, meta)
    loss.backward()
    optimizer.step()
    for (_, p0), (name, p1) in zip(initial_params, params):
        if name in whitelist:
            continue
        try:
            np.testing.assert_raises(AssertionError, torch.testing.assert_allclose, p0, p1)
        except AssertionError:
            if 'bias' in name:
                print('Warning: {} not updating'.format(name))
                continue
            if p1.grad.norm() > 1e-6:
                print('Warning: {} not significantly updating'.format(name))
                continue
            print('{} not updating'.format(name))
            for (nn1, pp1) in params:
                print('{} grad: {}'.format(nn1, pp1.grad.norm().item()))
            import pdb
            pdb.set_trace()
            raise


class TestBases(unittest.TestCase):
    def test_aj_i3d_updates(self):
        torch.manual_seed(12345)
        from models.bases import aj_i3d
        b, f, d = 2, 32, 224
        inputs = torch.randn(b, f, d, d, 3)
        args = Args()
        args.nclass = 157
        model = aj_i3d.AJ_I3D.get(args)
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
