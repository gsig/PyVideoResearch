"""
    Defines tasks for evaluation
"""
from misc_utils.utils import Timer
from models.bases.resnet50_3d_decoder import ResNet503DDecoder
from models.bases.resnet50_3d_decoder2 import ResNet503DDecoder2
from models.bases.resnet50_3d_decoder3 import ResNet503DDecoder3
from models.criteria.autoencoder_criterion import AutoencoderCriterion
from tasks.task import Task
from datasets.get import get_dataset
# from models.utils import set_distributed_backend
import torch
from datasets.utils import ffmpeg_video_writer
import copy


class AutoencoderTask(Task):
    def __init__(self, model, epoch, args):
        super(AutoencoderTask, self).__init__()
        self.num_videos = 1

    @classmethod
    def run(cls, model, criterion, epoch, args):
        task = cls(model, epoch, args)
        loader, = get_dataset(args, splits=('val', ), dataset=args.dataset)
        model = model.module
        model.eval()
        return task.stabilize_all(loader, model, epoch, args)

    def fine_tune_autoencoder(self, inputs, model, args):
        model = copy.deepcopy(model)
        model.train()
        params = model.parameters()
        #lr = 1e-4
        #warmup = 10
        #optimizer = torch.optim.SGD(params, lr=lr, momentum=args.momentum, weight_decay=0)
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=0)
        criteria = AutoencoderCriterion(args)
        tol = 1e-2
        loss = torch.Tensor([999])
        timer = Timer()
        try:
            with torch.enable_grad():
                num_iter = 0
                while loss > tol:
                    #if num_iter > warmup:
                    #    lr = 1e-3
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
        except KeyboardInterrupt as e:
            print(e)
        return model, x_hat

    def stabilize_all(self, loader, model, epoch, args):
        timer = Timer()
        for i, (inputs, target, meta) in enumerate(loader):
            if i >= self.num_videos:
                break
            if not args.cpu:
                inputs = inputs.cuda()
                target = target.cuda(async=True)
            original = inputs.detach().clone()
            reconstructed = model(inputs, None)[0]
            specific_model, fine_tuned = self.fine_tune_autoencoder(inputs, model, args)
            #fine_tuned = specific_model(inputs, None)[0]

            # prepare videos
            original = original[0]
            fine_tuned = fine_tuned[0]
            reconstructed = reconstructed[0]
            original *= torch.Tensor([0.229, 0.224, 0.225])[None, None, None, :].to(original.device)
            original += torch.Tensor([0.485, 0.456, 0.406])[None, None, None, :].to(original.device)
            fine_tuned *= torch.Tensor([0.229, 0.224, 0.225])[None, None, None, :].to(original.device)
            fine_tuned += torch.Tensor([0.485, 0.456, 0.406])[None, None, None, :].to(original.device)
            reconstructed *= torch.Tensor([0.229, 0.224, 0.225])[None, None, None, :].to(original.device)
            reconstructed += torch.Tensor([0.485, 0.456, 0.406])[None, None, None, :].to(original.device)

            # save video
            name = '{}_{}'.format(meta[0]['id'], meta[0]['time'])
            ffmpeg_video_writer(original.cpu(), '{}/{}_original.mp4'.format(args.cache, name))
            ffmpeg_video_writer(fine_tuned.cpu(), '{}/{}_finetuned.mp4'.format(args.cache, name))
            ffmpeg_video_writer(reconstructed.cpu(), '{}/{}_reconstructed.mp4'.format(args.cache, name))
            combined = torch.cat((original.cpu(), reconstructed.cpu(), fine_tuned.cpu()), 2)
            ffmpeg_video_writer(combined, '{}/{}_combined.mp4'.format(args.cache, name))

            timer.tic()
            print('Autoencoder: [{0}/{1}]\t'
                  'Time {timer.val:.3f} ({timer.avg:.3f})'.format(
                      i, self.num_videos, timer=timer))

        return {}
