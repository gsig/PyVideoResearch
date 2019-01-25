import unittest
import torch


class Args():
    pass


def test_model_updates(inputs, model, target, criterion, meta={}):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    optimizer.zero_grad()
    params = [np for np in model.named_parameters()]
    initial_params = [(name, p.clone()) for (name, p) in params]
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


class TestBases(unittest.TestCase):
    #def test_default_wrapper_updates(self):
    #    torch.manual_seed(12345)
    #    from models.bases import resnet50_3d
    #    from models.wrappers import default_wrapper
    #    from models.criteria import default_criterion
    #    args = Args()
    #    args.pretrained = False
    #    args.freeze_batchnorm = False
    #    model = resnet50_3d.ResNet503D.get(args)
    #    model = default_wrapper.DefaultWrapper(model, args)
    #    b, f, d = 2, 32, 224
    #    inputs = torch.randn(b, f, d, d, 3)
    #    target = torch.zeros(2, 400)
    #    args.balanceloss = False
    #    args.window_smooth = 0
    #    criterion = default_criterion.DefaultCriterion(args)
    #    test_model_updates(inputs, model, target, criterion)

    # def test_frcnn_wrapper_updates(self):
    #     if torch.cuda.is_available():
    #         torch.manual_seed(12345)
    #         from models.bases import aj_i3d
    #         from models.wrappers import frcnn_wrapper3
    #         from models.criteria import frcnn_criterion3
    #         args = Args()
    #         args.nclass = 81
    #         args.input_size = 224
    #         args.freeze_base = False
    #         args.freeze_head = False
    #         args.freeze_batchnorm = False
    #         model = aj_i3d.AJ_I3D.get(args)
    #         model = frcnn_wrapper3.FRCNNWrapper3(model, args)
    #         model = model.cuda()
    #         b, f, d = 1, 32, 224
    #         inputs = torch.randn(b, f, d, d, 3).cuda()
    #         meta = [{
    #             'boxes': torch.Tensor([[.25, .25, .75, .75]]),
    #             'labels': torch.Tensor([1]).long(),
    #             'id': 'asdf',
    #             'start': 1,
    #         }]
    #         target = None
    #         args.balanceloss = False
    #         args.window_smooth = 0
    #         criterion = frcnn_criterion3.FRCNNCriterion3(args)
    #         criterion = criterion.cuda()
    #         test_model_updates(inputs, model, target, criterion, meta)

    def test_maskrcnn_wrapper_updates(self):
        if torch.cuda.is_available():
            torch.manual_seed(12345)
            from models.bases import aj_i3d
            from models.wrappers import maskrcnn_wrapper
            from models.criteria import maskrcnn_criterion
            args = Args()
            args.nclass = 81
            args.input_size = 224
            args.freeze_base = False
            args.freeze_head = False
            args.freeze_batchnorm = False
            model = aj_i3d.AJ_I3D.get(args)
            model = maskrcnn_wrapper.MaskRCNNWrapper(model, args)
            model = model.cuda()
            b, f, d = 1, 32, 224
            inputs = torch.randn(b, f, d, d, 3).cuda()
            meta = [{
                'boxes': torch.Tensor([[.25, .25, .75, .75]]),
                'labels': torch.Tensor([1]).long(),
                'id': 'asdf',
                'start': 1,
            }]
            target = None
            args.balanceloss = False
            args.window_smooth = 0
            criterion = maskrcnn_criterion.MaskRCNNCriterion(args)
            criterion = criterion.cuda()
            test_model_updates(inputs, model, target, criterion, meta)
