import unittest
from models.test.test_bases import test_model_updates
from models.layers.nonlocal_block_3d import NONLocalBlock3D
import torch


class TestNonlocalBlock3D(unittest.TestCase):
    def test_nonlocal_block_3d_updates(self):
        torch.manual_seed(12345)
        b, f, d = 5, 512, 24
        inputs = torch.randn(b, f, d, d, d)
        model = NONLocalBlock3D(f)
        target = torch.zeros(b, f, d, d, d)
        whitelist = ['phi.0.bias', 'W.0.bias', 'theta.bias', 'g.0.bias']
        test_model_updates(inputs, model, target, whitelist)

    def test_nonlocal_block_3d_group_size_updates(self):
        torch.manual_seed(12345)
        b, f, d = 5, 512, 24
        inputs = torch.randn(b, f, d, d, d)
        model = NONLocalBlock3D(f, group_size=4)
        target = torch.zeros(b, f, d, d, d)
        whitelist = ['phi.0.bias', 'W.0.bias', 'theta.bias', 'g.0.bias']
        test_model_updates(inputs, model, target, whitelist)
