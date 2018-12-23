"""
    Defines tasks for evaluation
"""
from misc_utils.utils import Timer, AverageMeter
from models.wrappers.feature_extractor_wrapper import FeatureExtractorWrapper
from tasks.task import Task
from datasets.get import get_dataset
# from models.utils import set_distributed_backend
from collections import OrderedDict
import torch
import torch.nn.functional as F
from datasets.utils import ffmpeg_video_writer
from models.layers.video_stabilizer import VideoStabilizer
from misc_utils.video import video_trajectory, trajectory_loss


class StabilizationTask(Task):
    def __init__(self, model, epoch, args):
        super(StabilizationTask, self).__init__()
        self.num_videos = 50
        self.content_weight = args.content_weight
        self.motion_weight = args.motion_weight
        self.stabilization_target = args.stabilization_target

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
        if self.stabilization_target == 'video':
            params = [video.requires_grad_()]
        elif self.stabilization_target == 'transformer':
            transformer = VideoStabilizer(64).to(next(model.parameters()).device)
            params = transformer.parameters()
        else:
            assert False, "invalid stabilization target"
        optimizer = torch.optim.Adam(params,
                                     lr=args.lr, weight_decay=args.weight_decay)
        video_min, video_max = video.min().item(), video.max().item()
        target = model(video)
        target = OrderedDict((k, v.detach().clone()) for k, v in target.items())  # freeze targets
        timer = Timer()
        for num_iter in range(args.epochs):
            optimizer.zero_grad()
            if self.stabilization_target == 'video':
                video.data.clamp_(video_min, video_max)
                output = model(video)
                video_transformed = video
            elif self.stabilization_target == 'transformer':
                video_transformed = transformer(video)
                output = model(video_transformed)
            else:
                assert False, "invalid stabilization target"
            content_loss = F.mse_loss(output['fc'], target['fc'])
            # motion_loss = F.mse_loss(output['conv1'], target['conv1'].clone().zero_())
            motion_loss = F.l1_loss(video_transformed[:, 1:, :, :], video_transformed[:, :-1, :, :])
            loss = content_loss * self.content_weight + motion_loss * self.motion_weight
            loss.backward()
            optimizer.step()
            timer.tic()
            if num_iter % args.print_freq == 0:
                print('    Iter: [{0}/{1}]\t'
                      'Time {timer.val:.3f} ({timer.avg:.3f}) Content Loss: {2} \tMotion Loss: {3}'.format(
                          num_iter, args.epochs, content_loss.item(), motion_loss.item(), timer=timer))
        print('Stabilization Done')
        return video_transformed, content_loss.item(), motion_loss.item()

    def stabilize_all(self, loader, model, epoch, args):
        timer = Timer()
        content_losses = AverageMeter()
        motion_losses = AverageMeter()
        original_losses = AverageMeter()
        output_losses = AverageMeter()
        for i, (inputs, target, meta) in enumerate(loader):
            if i >= self.num_videos:
                break
            if not args.cpu:
                inputs = inputs.cuda()
                target = target.cuda(async=True)
            original = inputs.detach().clone()
            with torch.enable_grad():
                output, content_loss, motion_loss = self.stabilize_video(inputs, model, args)
            content_losses.update(content_loss)
            motion_losses.update(motion_loss)

            # prepare videos
            original = original[0]
            output = output[0]
            original *= torch.Tensor([0.229, 0.224, 0.225])[None, None, None, :].to(original.device)
            original += torch.Tensor([0.485, 0.456, 0.406])[None, None, None, :].to(original.device)
            output *= torch.Tensor([0.229, 0.224, 0.225])[None, None, None, :].to(output.device)
            output += torch.Tensor([0.485, 0.456, 0.406])[None, None, None, :].to(output.device)

            # save video
            name = '{}_{}'.format(meta[0]['id'], meta[0]['time'])
            ffmpeg_video_writer(original.cpu(), '{}/{}_original.mp4'.format(args.cache, name))
            ffmpeg_video_writer(output.cpu(), '{}/{}_stabilized.mp4'.format(args.cache, name))
            combined = torch.cat((original.cpu(), output.cpu()), 1)
            ffmpeg_video_writer(combined, '{}/{}_combined.mp4'.format(args.cache, name))

            # calculate stability losses
            print('calculating stability losses')
            original_trajectory = video_trajectory(original)
            original_losses.update(trajectory_loss(original_trajectory))
            output_trajectory = video_trajectory(output)
            output_losses.update(trajectory_loss(output_trajectory))
            timer.tic()
            print('Stabilization: [{0}/{1}]\t'
                  'Time {timer.val:.3f} ({timer.avg:.3f})'.format(
                      i, len(loader), timer=timer))

        scores = {'stabilization_task_content_loss': content_losses.avg,
                  'stabilization_task_motion_loss': motion_losses.avg,
                  'stabilization_task_original_loss': original_losses.avg,
                  'stabilization_task_output_loss': output_losses.avg}
        return scores
