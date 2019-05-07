from __future__ import absolute_import

from models.wrappers.wrapper import Wrapper
import torch
import sys
import os.path
import os
#import scipy
import numpy as np
from argparse import Namespace
#from models.bases.aj_i3d import Unit3D
#from collections import defaultdict
#import torch.nn.functional as F
#import torch.nn.init as init


def str2torch(str_array):
    return torch.Tensor(np.fromstring(str_array, dtype=np.uint8))


class MaskRCNNWrapper(Wrapper):
    def init_maskrcnn_benchmark(self, args):
        if 'datasets' in sys.modules:
            del sys.modules['datasets']
        this_dir = os.path.dirname(__file__)
        lib_path = os.path.join(this_dir, '../../external/maskrcnn-benchmark')
        sys.path.insert(0, lib_path)

        # config
        from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
        from maskrcnn_benchmark.config import cfg
        self.cfg = cfg
        #cfg.MODEL.BACKBONE.OUT_CHANNELS = 832
        #cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 208
        cfg.MODEL.RPN.PRE_NMS_TOP_N_TEST = 6000
        cfg.MODEL.RPN.POST_NMS_TOP_N_TEST = 1000
        cfg.MODEL.ROI_HEADS.SCORE_THRESH = 0
        cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 0

        # bind classes/methods from MaskRCNN-Benchmark
        from maskrcnn_benchmark.modeling.rpn.rpn import build_rpn
        from maskrcnn_benchmark.modeling.roi_heads.roi_heads import build_roi_heads
        from maskrcnn_benchmark.structures.image_list import to_image_list
        from maskrcnn_benchmark.structures.bounding_box import BoxList
        from maskrcnn_benchmark.modeling.backbone import build_backbone

        self.basenet = build_backbone(cfg)
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)
        self.to_image_list = to_image_list
        self.BoxList = BoxList

    def __init__(self, basenet, args):
        super(MaskRCNNWrapper, self).__init__(basenet, args)
        #self.basenet = basenet
        self.init_maskrcnn_benchmark(args)
        self.freeze_batchnorm = args.freeze_batchnorm
        self.n_class = args.nclass
        self.input_size = args.input_size
        self.freeze_base = args.freeze_base
        self.freeze_head = args.freeze_head
        
        # for visualizing bounding boxes
        # this_dir = os.path.dirname(__file__)
        # lib_path = os.path.join(this_dir, '../../external/Detectron.pytorch/lib')
        # sys.path.insert(0, lib_path)
        # import utils.vis as vis_utils
        # self.vis_utils = vis_utils
        # sys.path.pop(0)

        # for full i3d model
        #for i, end_point in enumerate(self.basenet.VALID_ENDPOINTS):
        #    if end_point == 'Mixed_4f':  # first half should include Mixed_4f
        #        self.first_layers = self.basenet.VALID_ENDPOINTS[:i+1]
        #        self.last_layers = self.basenet.VALID_ENDPOINTS[i+1:]

    def baseforward(self, x, part):
        layers = self.first_layers if part == 'first' else self.last_layers
        for end_point in layers:
            if end_point in self.basenet.end_points:
                x = self.basenet._modules[end_point](x)
        return x

    def forward(self, im, meta):
        if self.freeze_batchnorm:
            for module in self.basenet.modules():
                if isinstance(module, torch.nn.modules.BatchNorm1d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm2d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm3d):
                    module.eval()

        # x is of the form b x n x h x w x c
        # model expects b x c x n x h x w
        #x = im.permute(0, 4, 1, 2, 3)
        #img_size = x.shape[3:]
        #with torch.set_grad_enabled(not self.freeze_base):
        #    x = self.baseforward(x, 'first')

        ## slice feature map to get center frame
        #t = x.shape[2]
        #x_slice = x[:, :, t//2, :, :]
        #features = [x_slice]

        x = im.permute(0, 4, 1, 2, 3)
        img_size = x.shape[3:]
        t = x.shape[2]
        x_slice = x[:, :, t//2, :, :]
        images = self.to_image_list(x_slice)
        features = self.basenet(images.tensors)

        # pass through region proposal network
        images = Namespace(image_sizes=[(self.input_size, self.input_size)] * x.shape[0])
        targets = [self.BoxList(m['boxes'].cuda() * self.input_size, (self.input_size, self.input_size)) for m in meta]
        for m, box in zip(meta, targets):
            box.add_field('labels', m['labels'].cuda() + 1)  # 0 is bg class
        proposals, proposal_losses = self.rpn(images, features, targets)
        x, result, detector_losses = self.roi_heads(features, proposals, targets)

        # we want predictions even at training time
        # x, result, detector_losses = self.roi_heads(features, proposals, targets)
        # class_logits, box_regression = self.roi_heads.box.predictor(x)
        # result = self.roi_heads.box.post_processor((class_logits, box_regression), proposals)

        # get bounding boxes for prediction
        score_predictions = []
        for i, (m, pred_boxlist) in enumerate(zip(meta, result)):
            try:
                scores = pred_boxlist.get_field("scores")
            except Exception:
                scores = pred_boxlist.get_field("labels")
            score_prediction = {'do_not_collate': True,
                                'vid': str2torch(m['id']),
                                'start': torch.Tensor([(m['start'])]),
                                'boxes': pred_boxlist.bbox.cpu() / img_size[0],
                                'labels': pred_boxlist.get_field("labels").cpu() - 1,  # 0 is bg class
                                'scores': scores.cpu()}
            score_predictions.append(score_prediction)

        # visualization
        if False:
            cls_boxes = [[] for _ in range(81)]
            score_prediction = score_predictions[-1]
            boxes = score_prediction['boxes']
            labels = score_prediction['labels']
            scores = score_prediction['scores']
            for box, label, score in zip(boxes, labels, scores):
                cls_boxes[label].append(torch.cat((box*img_size[0], score.view(-1).float())))
            cls_boxes = [[] if x == [] else torch.stack(x).numpy() for x in cls_boxes]
            import types
            dataset = types.SimpleNamespace()
            setattr(dataset, 'classes', [str(cls) for cls in range(81)])
            self.vis_utils.vis_one_image(
                im[-1, im.shape[1]//2, :, :].cpu().numpy()/2 + 1/2,
                'visual',
                './',
                cls_boxes,
                thresh=.05,
                box_alpha=0.8,
                dataset=dataset,
                show_class=True
            )
            import pdb
            pdb.set_trace()

        return score_predictions, proposal_losses, detector_losses
