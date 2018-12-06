""" Defines the Trainer class which handles train/validation/validation_video
"""
import torch
import itertools
from misc_utils.utils import AverageMeter, submission_file, Timer
from torch.nn.parallel.scatter_gather import gather


def adjust_learning_rate(startlr, decay_rate, optimizer, epoch):
    """ TODO """
    if type(decay_rate) == int:
        decay_rate = '{}'.format(decay_rate)
    if ',' not in decay_rate:
        decay_rate = int(decay_rate)
        decay_rate = '{},{},{}'.format(decay_rate, 2*decay_rate, 3*decay_rate)
    decay_rates = [int(x) for x in decay_rate.split(',')]
    lr = startlr
    for d in decay_rates:
        if epoch >= d:
            lr *= 0.1
    print('lr = {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def part(x, iter_size):
    n = int(len(x)*iter_size)
    if iter_size > 1.0:
        x = itertools.chain.from_iterable(itertools.repeat(x))
    return itertools.islice(x, n)


class Trainer(object):
    def train(self, loader, model, criterion, optimizer, epoch, metrics, args, validate=False):
        timer = Timer()
        data_time = AverageMeter()
        losses = AverageMeter()
        metrics = [m() for m in metrics]

        if validate:
            # switch to evaluate mode
            model.eval()
            criterion.eval()
            iter_size = args.val_size
            setting = 'Validate Epoch'
        else:
            # switch to train mode
            adjust_learning_rate(args.lr, args.lr_decay_rate, optimizer, epoch)
            model.train()
            criterion.train()
            optimizer.zero_grad()
            iter_size = args.train_size
            setting = 'Train Epoch'

        for i, (input, target, meta) in enumerate(part(loader, iter_size)):
            if args.synchronous:
                assert meta['id'][0] == meta['id'][1], "dataset not synced"
            data_time.update(timer.thetime() - timer.end)

            if not args.cpu:
                target = target.cuda(async=True)
            output = model(input, meta)
            if type(output) != tuple:
                output = (output,)
            scores, loss, score_target = criterion(*(output + (target, meta)))
            losses.update(loss.item())
            with torch.no_grad():
                for m in metrics:
                    m.update(scores, score_target)

            if not validate:
                loss.backward()
                if i % args.accum_grad == args.accum_grad-1:
                    print('updating parameters')
                    optimizer.step()
                    optimizer.zero_grad()

            timer.tic()
            if i % args.print_freq == 0:
                print('[{name}] {setting}: [{0}][{1}/{2}({3})]\t'
                      'Time {timer.val:.3f} ({timer.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      '{metrics}'.format(
                          epoch, i, int(len(loader)*iter_size), len(loader),
                          name=args.name, setting=setting, timer=timer,
                          data_time=data_time, loss=losses,
                          metrics=' \t'.join(str(m) for m in metrics)))
            del loss, output, target  # make sure we don't hold on to the graph

        metrics = dict(m.compute() for m in metrics)
        metrics.update({'loss_': losses.avg})
        metrics = dict((k+'val', v) if validate else (k+'train', v) for k, v in metrics.items())
        return metrics

    def validate(self, loader, model, criterion, epoch, metrics, args):
        """
            Validate in the same approach as training
        """
        with torch.no_grad():
            return self.train(loader, model, criterion, None, epoch, metrics, args, validate=True)

    def validate_video(self, loader, model, criterion, epoch, metrics, args):
        """ Run video-level validation on the test set """
        with torch.no_grad():
            timer = Timer()
            ids, outputs = [], []
            metrics = [m() for m in metrics]

            # switch to evaluate mode
            model.eval()
            criterion.eval()

            for i, (input, target, meta) in enumerate(loader):
                if not args.cpu:
                    target = target.cuda(async=True)

                # split batch into smaller chunks
                if args.video_batch_size == -1:
                    output = model(input, meta)
                else:
                    output_chunks = []
                    for chunk in input.split(args.video_batch_size):
                        output_chunks.append(model(chunk, meta))
                    #if type(output_chunks[0]) == tuple:
                    #    output = tuple(torch.cat(x) for x in zip(*output_chunks))
                    #else:
                    #    output = torch.cat(output_chunks)
                    output = gather(output_chunks)

                if type(output) != tuple:
                    output = (output,)
                scores, loss, score_target = criterion(*(output + (target, meta)), synchronous=True)
                for m in metrics:
                    m.update(scores, score_target)

                # store predictions
                scores_video = scores.max(dim=0)[0]
                outputs.append(scores_video.cpu())
                timer.tic()
                if i % args.print_freq == 0:
                    print('[{name}] ValidateVideo: [{0}/{1}]\t'
                          'Time {timer.val:.3f} ({timer.avg:.3f})\t'
                          '{metrics}'.format(
                              i, len(loader), timer=timer, name=args.name,
                              metrics=' \t'.join(str(m) for m in metrics)))
            submission_file(
                ids, outputs, '{}/epoch_{:03d}.txt'.format(args.cache, epoch+1))
            metrics = dict(m.compute() for m in metrics)
            metrics = dict((k+'valvideo', v) for k, v in metrics.items())
            print(metrics)
            return metrics
