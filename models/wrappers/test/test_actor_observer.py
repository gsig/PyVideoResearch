import unittest
import torch


class Args():
    pass


def test_model_updates(inputs, model, target, criterion, meta={}):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    optimizer.zero_grad()
    params = [np for np in model.named_parameters()]
    initial_params = [(name, p.clone()) for (name, p) in params]
    for _ in range(2):
        output = model(inputs, meta)
        if type(output) is not tuple:
            output = (output, )
        _, loss, _ = criterion(*(output + (target, meta)))
        loss.backward()
        optimizer.step()
    for (_, p0), (name, p1) in zip(initial_params, params):
        try:
            assert not torch.equal(p0, p1)
        except AssertionError:
            print(name)
            raise


class TestActorObserver(unittest.TestCase):
    def test_actor_observer_with_classifier_updates(self):
        torch.manual_seed(12345)
        from torchvision.models.resnet import resnet50
        from models.wrappers.actor_observer_with_classifier_wrapper import ActorObserverWithClassifierWrapper
        from models.criteria.actor_observer_with_classifier_criterion import ActorObserverWithClassifierCriterion
        args = Args()
        args.nclass = 157
        args.freeze_batchnorm = False
        args.finaldecay = 0.9
        args.decay = 0.9
        args.margin = 0.0
        args.classifier_weight = 1.0
        args.share_selector = False
        args.normalize_per_video = False
        model = resnet50()
        model = ActorObserverWithClassifierWrapper(model, args)
        b, d = 10, 224
        inputs = [torch.randn(b, 3, d, d), torch.randn(b, 3, d, d), torch.randn(b, 3, d, d)]
        meta = {'thirdtime': torch.zeros(b),
                'firsttime_pos': torch.zeros(b),
                'firsttime_neg': torch.zeros(b),
                'n': torch.zeros(b),
                'n_ego': torch.zeros(b),
                'id': ['asdf'] * b,
                }
        target = torch.ones(b, args.nclass)
        target[b//2:, 0] = -1
        args.balanceloss = False
        args.window_smooth = 0
        criterion = ActorObserverWithClassifierCriterion(args)
        test_model_updates(inputs, model, target, criterion, meta)

    def test_actor_observer_with_classifier_3d_updates(self):
        torch.manual_seed(12345)
        from models.bases.resnet50_3d import ResNet503D
        from models.wrappers.actor_observer_with_classifier_wrapper import ActorObserverWithClassifierWrapper
        from models.criteria.actor_observer_with_classifier_criterion import ActorObserverWithClassifierCriterion
        args = Args()
        args.nclass = 157
        args.pretrained = False
        args.freeze_batchnorm = False
        args.finaldecay = 0.9
        args.decay = 0.9
        args.margin = 0.0
        args.classifier_weight = 1.0
        args.share_selector = False
        args.normalize_per_video = False
        model = ResNet503D.get(args)
        model = ActorObserverWithClassifierWrapper(model, args)
        b, f, d = 2, 16, 224
        inputs = [torch.randn(b, f, d, d, 3), torch.randn(b, f, d, d, 3), torch.randn(b, f, d, d, 3)]
        meta = {'thirdtime': torch.zeros(b),
                'firsttime_pos': torch.zeros(b),
                'firsttime_neg': torch.zeros(b),
                'n': torch.zeros(b),
                'n_ego': torch.zeros(b),
                'id': ['asdf'] * b,
                }
        target = torch.ones(b, args.nclass)
        target[b//2:, 0] = -1
        args.balanceloss = False
        args.window_smooth = 0
        criterion = ActorObserverWithClassifierCriterion(args)
        test_model_updates(inputs, model, target, criterion, meta)
