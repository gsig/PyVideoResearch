""" Defines the Trainer class which handles train/validation/validation_video
"""
import torch
import itertools
from misc_utils.utils import AverageMeter, Timer


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
        metrics.update({'loss': losses.avg})
        metrics = dict(('val_'+k, v) if validate else ('train_'+k, v) for k, v in metrics.items())
        return metrics

    def validate(self, loader, model, criterion, epoch, metrics, args):
        """
            Validate in the same approach as training
        """
        with torch.no_grad():
            return self.train(loader, model, criterion, None, epoch, metrics, args, validate=True)
