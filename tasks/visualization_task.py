"""
    Defines tasks for evaluation
"""
from tasks.task import Task
from datasets.get import get_dataset
import torch
from datasets.utils import ffmpeg_video_writer


class VisualizationTask(Task):
    def __init__(self, model, epoch, args):
        super(VisualizationTask, self).__init__()
        self.num_videos = 5

    @classmethod
    def run(cls, model, criterion, epoch, args):
        task = cls(model, epoch, args)
        train_loader, val_loader = get_dataset(args, splits=('train', 'val'), dataset=args.dataset)
        model.eval()
        task.visualize_all(train_loader, model, epoch, args, 'train')
        task.visualize_all(val_loader, model, epoch, args, 'val')
        return {'visualization_task': args.cache}

    def visualize_all(self, loader, model, epoch, args, split):
        for i, (inputs, target, meta) in enumerate(loader):
            if i >= self.num_videos:
                break
            if not args.cpu:
                inputs = inputs.cuda()
                target = target.cuda(async=True)
            x_hat, code, x = model(inputs, args)

            # prepare videos
            original = x[0]
            output = x_hat[0]
            original *= torch.Tensor([0.229, 0.224, 0.225])[None, None, None, :].to(original.device)
            original += torch.Tensor([0.485, 0.456, 0.406])[None, None, None, :].to(original.device)
            output *= torch.Tensor([0.229, 0.224, 0.225])[None, None, None, :].to(output.device)
            output += torch.Tensor([0.485, 0.456, 0.406])[None, None, None, :].to(output.device)

            # save video
            name = '{}_{:03d}_{}_{}'.format(split, epoch, meta[0]['id'], meta[0]['time'])
            ffmpeg_video_writer(original.cpu(), '{}/{}_original.mp4'.format(args.cache, name))
            ffmpeg_video_writer(output.cpu(), '{}/{}_output.mp4'.format(args.cache, name))
            combined = torch.cat((original.cpu(), output.cpu()), 2)
            ffmpeg_video_writer(combined, '{}/{}_combined.mp4'.format(args.cache, name))

            print('Visualization: [{0}/{1}]'.format(i, self.num_videos))
