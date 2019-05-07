from misc_utils.utils import submission_file, Timer
from torch.nn.parallel.scatter_gather import gather
from metrics.get import get_metrics
from datasets.get import get_dataset
from tasks.task import Task


class VideoTask(Task):
    def __init__(self, model, epoch, args):
        super(VideoTask, self).__init__()
        self.metrics = get_metrics(args.video_metrics)

    @classmethod
    def run(cls, model, criterion, epoch, args):
        task = cls(model, epoch, args)
        loader, = get_dataset(args, splits=('val_video', ))
        return task.validate_video(loader, model, criterion, epoch, args)

    def validate_video(self, loader, model, criterion, epoch, args):
        """ Run video-level validation on the test set """
        timer = Timer()
        ids, outputs = [], []
        metrics = [m() for m in self.metrics]

        # switch to evaluate mode
        model.eval()
        criterion.eval()

        for i, (input, target, meta) in enumerate(loader):
            if not args.cpu:
                input = input.cuda()
                target = target.cuda(async=True)

            # split batch into smaller chunks
            if args.video_batch_size == -1:
                output = model(input, meta)
            else:
                output_chunks = []
                for chunk in input.split(args.video_batch_size):
                    output_chunks.append(model(chunk, meta))
                output = gather(output_chunks, input.device)

            if type(output) != tuple:
                output = (output,)
            scores, loss, score_target = criterion(*(output + (target, meta)), synchronous=True)
            for m in metrics:
                m.update(scores, score_target)

            # store predictions
            scores_video = scores.max(dim=0)[0]
            outputs.append(scores_video.cpu())
            # ids.append(meta['id'][0])
            timer.tic()
            if i % args.print_freq == 0:
                print('[{name}] {task}: [{0}/{1}]\t'
                      'Time {timer.val:.3f} ({timer.avg:.3f})\t'
                      '{metrics}'.format(
                          i, len(loader), timer=timer, name=args.name, task=self.name,
                          metrics=' \t'.join(str(m) for m in metrics)))
            del loss, output, target  # make sure we don't hold on to the graph
        submission_file(
            ids, outputs, '{}/epoch_{:03d}.txt'.format(args.cache, epoch+1))
        metrics = dict(m.compute() for m in metrics)
        metrics = dict((self.name+'_'+k, v) for k, v in metrics.items())
        print(metrics)
        return metrics
