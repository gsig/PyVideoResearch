import unittest
from models.test.test_bases import test_model_updates
from models.layers.nonlocal_block_3d import NONLocalBlock3D
import torch


class TestNonlocalBlock3D(unittest.TestCase):
    def test_nonlocal_block_3d_updates(self):
        torch.manual_seed(12345)
        b, f, d = 2, 100, 2
        inputs = torch.randn(b, f, d, d, d)
        model = NONLocalBlock3D(100)
        model = torch.nn.Sequential(model, torch.nn.MaxPool3d(2))
        target = torch.zeros(2, 100)
        test_model_updates(inputs, model, target)

    def test_nonlocal_block_3d_group_size_updates(self):
        torch.manual_seed(12345)
        b, f, d = 2, 100, 2
        inputs = torch.randn(b, f, d, d, d)
        model = NONLocalBlock3D(100, group_size=4)
        model = torch.nn.Sequential(model, torch.nn.MaxPool3d(2))
        target = torch.zeros(2, 100)
        test_model_updates(inputs, model, target)
