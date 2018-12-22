"""
    Defines tasks for evaluation
"""
from misc_utils.utils import Timer
from models.wrappers.feature_extractor_wrapper import FeatureExtractorWrapper
from tasks.task import Task
from datasets.get import get_dataset
# from models.utils import set_distributed_backend
from collections import OrderedDict
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils import ffmpeg_video_writer


class StabilizationTask(Task):
    def __init__(self, model, epoch, args):
        super(StabilizationTask, self).__init__()
        self.num_align = 1
        self.content_weight = 1
        self.motion_weight = 1

    @classmethod
    def run(cls, model, criterion, epoch, args):
        task = cls(model, epoch, args)
        loader, = get_dataset(args, splits=('val', ), dataset=args.dataset)
        model = FeatureExtractorWrapper(model, args)
        # model = set_distributed_backend(model, args)
        model.eval()
        return task.stabilize_all(loader, model, epoch, args)

    def stabilize_video(self, video, model, args):
        # optimizer = torch.optim.LBFGS([video.requires_grad_()])
        optimizer = torch.optim.Adam([video.requires_grad_()],
                                     lr=args.lr, weight_decay=args.weight_decay)
        video_min, video_max = video.min().item(), video.max().item()
        target = model(video)
        target = OrderedDict((k, v.detach().clone()) for k, v in target.items())  # freeze targets
        timer = Timer()
        for num_iter in range(args.epochs):
            optimizer.zero_grad()
            video.data.clamp_(video_min, video_max)
            output = model(video)
            content_loss = F.mse_loss(output['fc'], target['fc'])
            motion_loss = F.mse_loss(output['conv1'], target['conv1'].zero())
            # motion_loss = F.l1_loss(video[:, 1:, :, :], video[:, :-1, :, :])
            loss = content_loss * self.content_weight + motion_loss * self.motion_weight
            loss.backward()
            optimizer.step()
            timer.tic()
            if num_iter % args.print_freq == 0:
                print('    Iter: [{0}/{1}]\t'
                      'Time {timer.val:.3f} ({timer.avg:.3f}) Content Loss: {2} \tMotion Loss: {3}'.format(
                          num_iter, args.epochs, content_loss.item(), motion_loss.item(), timer=timer))
            import pdb
            pdb.set_trace()
        print('Stabilization Done')
        return video

    def stabilize_all(self, loader, model, epoch, args):
        timer = Timer()
        unnormalize = transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                           std=[1/0.229, 1/0.224, 1/0.225])
        for i, (inputs, target, meta) in enumerate(loader):
            if i >= self.num_align:
                break
            if not args.cpu:
                inputs = inputs.cuda()
                target = target.cuda(async=True)
            original = inputs.detach().clone()
            with torch.enable_grad():
                output = self.stabilize_video(inputs, model, args)
            original = unnormalize(original)
            output = unnormalize(output)
            ffmpeg_video_writer(original, args.cache + '/original.mp4')
            ffmpeg_video_writer(output, args.cache + '/stabilized.mp4')
            timer.tic()
            print('Stabilization: [{0}/{1}]\t'
                  'Time {timer.val:.3f} ({timer.avg:.3f})'.format(
                      i, len(loader), timer=timer))
            import pdb
            pdb.set_trace()

        scores = {}  # TODO
        return scores
