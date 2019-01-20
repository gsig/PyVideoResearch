"""
    Defines tasks for evaluation
"""
from misc_utils.utils import Timer, AverageMeter
from models.wrappers.feature_extractor_wrapper import FeatureExtractorWrapper
from models.bases.resnet50_3d_decoder import ResNet503DDecoder
from models.bases.resnet50_3d_decoder2 import ResNet503DDecoder2
from models.bases.resnet50_3d_decoder3 import ResNet503DDecoder3
from models.criteria.autoencoder_criterion import AutoencoderCriterion
from tasks.task import Task
from datasets.get import get_dataset
# from models.utils import set_distributed_backend
from collections import OrderedDict
import torch
import torch.nn.functional as F
from datasets.utils import ffmpeg_video_writer
from models.layers.video_stabilizer import VideoStabilizer
from misc_utils.video import video_trajectory, trajectory_loss
import random
import math
import copy


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


class StabilizationAutoencoderTask(Task):
    def __init__(self, model, epoch, args):
        super(StabilizationAutoencoderTask, self).__init__()
        self.num_videos = 50
        self.content_weight = args.content_weight
        self.motion_weight = args.motion_weight
        self.style_weight = args.style_weight
        self.grid_weight = args.grid_weight
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
        # model = set_distributed_backend(model, args)
        model = model.module
        model.eval()
        return task.stabilize_all(loader, model, epoch, args)

    def fine_tune_autoencoder(self, inputs, model, args):
        model = copy.deepcopy(model)
        params = model.parameters()
        lr = 1e-5
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0)
        criteria = AutoencoderCriterion(args)
        tol = 1e-2
        loss = torch.Tensor([999])
        num_iter = 0
        timer = Timer()
        with torch.enable_grad():
            while loss > tol:
                optimizer.zero_grad()
                x_hat, code, x = model(inputs, None)
                _, loss, _ = criteria(x_hat, code, x, None, None)
                loss.backward()
                optimizer.step()
                num_iter += 1
                timer.tic()
                if num_iter % args.print_freq == 0:
                    print('    Iter: [{0}]\t'
                          'Time {timer.val:.3f} ({timer.avg:.3f}) Loss: {1}'.format(
                              num_iter, loss, timer=timer))
        return model

    def stabilize_video(self, video, model, args):
        if self.stabilization_target == 'autoencoder':
            model = model.basenet.decoder
            params = model.parameters()
        else:
            assert False, "invalid stabilization target"

        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        n_params = [x.numel() for x in params]
        original_params = [v.detach().clone() for v in model.parameters()]
        x_hat, code, x = model(video)
        timer = Timer()
        for num_iter in range(args.epochs):
            optimizer.zero_grad()

            if self.stabilization_target == 'autoencoder':
                video_transformed = model(code)
            else:
                assert False, "invalid stabilization target"

            content_loss = [((a - b)**2).sum() for a, b in zip(params, original_params)]
            content_loss = (sum(content_loss) / n_params).sqrt()
            motion_loss = F.l1_loss(video_transformed[:, 1:, :, :, :], video_transformed[:, :-1, :, :, :])

            loss = (content_loss * self.content_weight +
                    motion_loss * self.motion_weight)
            loss.backward()
            optimizer.step()
            timer.tic()
            if num_iter % args.print_freq == 0:
                print('    Iter: [{0}/{1}]\t'
                      'Time {timer.val:.3f} ({timer.avg:.3f}) '
                      'Content Loss: {2} \tMotion Loss: {3}\t Style Loss: {4}'.format(
                          num_iter, args.epochs,
                          content_loss.item(), motion_loss.item(), timer=timer))
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
            reconstructed = model(inputs, None)[0]
            specific_model = self.fine_tune_autoencoder(inputs, model, args)
            fine_tuned = specific_model(inputs, None)[0]
            with torch.enable_grad():
                output, content_loss, motion_loss = self.stabilize_video(inputs, specific_model, args)
            content_losses.update(content_loss)
            motion_losses.update(motion_loss)

            # prepare videos
            original = original[0]
            output = output[0]
            fine_tuned = fine_tuned[0]
            reconstructed = reconstructed[0]
            original *= torch.Tensor([0.229, 0.224, 0.225])[None, None, None, :].to(original.device)
            original += torch.Tensor([0.485, 0.456, 0.406])[None, None, None, :].to(original.device)
            output *= torch.Tensor([0.229, 0.224, 0.225])[None, None, None, :].to(output.device)
            output += torch.Tensor([0.485, 0.456, 0.406])[None, None, None, :].to(output.device)
            fine_tuned *= torch.Tensor([0.229, 0.224, 0.225])[None, None, None, :].to(output.device)
            fine_tuned += torch.Tensor([0.485, 0.456, 0.406])[None, None, None, :].to(output.device)
            reconstructed *= torch.Tensor([0.229, 0.224, 0.225])[None, None, None, :].to(output.device)
            reconstructed += torch.Tensor([0.485, 0.456, 0.406])[None, None, None, :].to(output.device)

            # save video
            name = '{}_{}'.format(meta[0]['id'], meta[0]['time'])
            ffmpeg_video_writer(original.cpu(), '{}/{}_original.mp4'.format(args.cache, name))
            ffmpeg_video_writer(output.cpu(), '{}/{}_stabilized.mp4'.format(args.cache, name))
            ffmpeg_video_writer(fine_tuned.cpu(), '{}/{}_finetuned.mp4'.format(args.cache, name))
            ffmpeg_video_writer(reconstructed.cpu(), '{}/{}_reconstructed.mp4'.format(args.cache, name))
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
