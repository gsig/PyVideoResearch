import unittest
import torch
from tasks.stabilization_task import StabilizationTask
from models.wrappers.feature_extractor_wrapper import FeatureExtractorWrapper


class Args():
    pass


class TestStabilizationTask(unittest.TestCase):
    def test_stabilization_task_with_single_iter(self):
        torch.manual_seed(12345)
        from models.bases.resnet50_3d import ResNet503D
        args = Args()
        args.pretrained = False
        model = ResNet503D.get(args)
        task = StabilizationTask(model, 0, args)
        args.features = 'conv1;fc'
        model = FeatureExtractorWrapper(model, args)
        b, f, d = 1, 16, 224
        inputs = torch.randn(b, f, d, d, 3)
        args.epochs = 1
        args.print_freq = 1
        args.lr = 0.1
        args.weight_decay = 1e-4
        task.stabilize_video(inputs, model, args)
