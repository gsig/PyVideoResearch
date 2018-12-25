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
from models.layers.video_deformer import VideoDeformer
from misc_utils.video import video_trajectory, trajectory_loss
import random
import math


def gram_matrix(x):
    conv3d = x.dim() == 5
    if conv3d:
        b, n, d1, d2, c = x.shape
        x = x.reshape(-1, d1, d2, c)
        x = x.contiguous()
        #x = x.permute(0, 3, 1, 2)

    a, b, c, d = x.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = x.view(a * b, c * d)  # resise F_XL into \hat F_XL

    g = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return g.div(a * b * c * d)


class StabilizationTask(Task):
    def __init__(self, model, epoch, args):
        super(StabilizationTask, self).__init__()
        self.num_videos = 50
        self.content_weight = args.content_weight
        self.motion_weight = args.motion_weight
        self.style_weight = args.style_weight
        self.stabilization_target = args.stabilization_target

    def augmentation(self, video):
        # https://distill.pub/2017/feature-visualization/
        channels = video.shape[1]
        augmenter = VideoStabilizer(channels).to(video.device)
        transform = torch.Tensor([1, 0, 0, 0, 1, 0]).float()

        # jittering
        transform[2] = random.randint(-16, 16) / 224.
        transform[5] = random.randint(-16, 16) / 224.

        # scaling
        scale = random.choice((1, 0.975, 1.025, 0.95, 1.05))
        transform[0] *= scale
        transform[4] *= scale

        # rotation
        rotation = random.choice((-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)) / 360. * 2 * math.pi
        transform[0] *= math.cos(rotation)
        transform[4] *= math.cos(rotation)
        transform[1] = -scale * math.sin(rotation)
        transform[3] = scale * math.sin(rotation)

        augmenter.theta.data.copy_(transform[None, :].repeat(channels, 1))
        return augmenter(video)

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
        elif self.stabilization_target == 'deformer':
            transformer = VideoDeformer(64).to(next(model.parameters()).device)
            params = transformer.parameters()
        elif self.stabilization_target == 'videotransformer':
            params = [video.requires_grad_()]
            transformer = VideoStabilizer(64).to(next(model.parameters()).device)
            params += list(transformer.parameters())
        elif self.stabilization_target == 'sum':
            original_video = video.clone()
            params = [video.requires_grad_()]
            transformer = VideoStabilizer(64).to(next(model.parameters()).device)
            params += list(transformer.parameters())
        else:
            assert False, "invalid stabilization target"
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        video_min, video_max = video.min().item(), video.max().item()
        target = model(video)
        target = OrderedDict((k, v.detach().clone()) for k, v in target.items())  # freeze targets
        timer = Timer()
        for num_iter in range(args.epochs):
            optimizer.zero_grad()
            if self.stabilization_target == 'video':
                video.data.clamp_(video_min, video_max)
                output = model(self.augmentation(video))
                video_transformed = video
            elif self.stabilization_target == 'transformer':
                video_transformed = transformer(video)
                output = model(video_transformed)
            elif self.stabilization_target == 'deformer':
                video_transformed = transformer(video)
                output = model(video_transformed)
            elif self.stabilization_target == 'videotransformer':
                video.data.clamp_(video_min, video_max)
                video_transformed = transformer(video)
                output = model(self.augmentation(video_transformed))
            elif self.stabilization_target == 'sum':
                video.data.clamp_(video_min, video_max)
                video_transformed = transformer(original_video)
                video_transformed += video
                output = model(self.augmentation(video_transformed))
            else:
                assert False, "invalid stabilization target"
            content_loss = F.mse_loss(output['fc'], target['fc'])
            style_loss = F.mse_loss(gram_matrix(output['layer1']), gram_matrix(target['layer1']))
            #motion_loss = F.l1_loss(video_transformed[:, 1:, :, :], video_transformed[:, :-1, :, :])
            motion_loss = F.mse_loss(video_transformed[:, 1:, :, :], video_transformed[:, :-1, :, :])
            loss = (content_loss * self.content_weight +
                    motion_loss * self.motion_weight +
                    style_loss * self.style_weight)
            loss.backward()
            optimizer.step()
            timer.tic()
            if num_iter % args.print_freq == 0:
                print('    Iter: [{0}/{1}]\t'
                      'Time {timer.val:.3f} ({timer.avg:.3f}) '
                      'Content Loss: {2} \tMotion Loss: {3}\t Style Loss: {4}'.format(
                          num_iter, args.epochs,
                          content_loss.item(), motion_loss.item(), style_loss.item(), timer=timer))
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
            combined = torch.cat((original.cpu(), output.cpu()), 2)
            ffmpeg_video_writer(combined, '{}/{}_combined.mp4'.format(args.cache, name))

            # calculate stability losses
            print('calculating stability losses')
            try:
                # this can fail when there are no feature matches found
                original_trajectory = video_trajectory(original.cpu().numpy())
                original_losses.update(trajectory_loss(original_trajectory))
                output_trajectory = video_trajectory(output.cpu().numpy())
                output_losses.update(trajectory_loss(output_trajectory))
            except Exception as e:
                print(e)
            timer.tic()
            print('Stabilization: [{0}/{1}]\t'
                  'Time {timer.val:.3f} ({timer.avg:.3f}) Original Loss {2} \t Output Loss {3}'.format(
                      i, self.num_videos, original_losses.avg, output_losses.avg, timer=timer))

        scores = {'stabilization_task_content_loss': content_losses.avg,
                  'stabilization_task_motion_loss': motion_losses.avg,
                  'stabilization_task_original_loss': original_losses.avg,
                  'stabilization_task_output_loss': output_losses.avg}
        return scores
