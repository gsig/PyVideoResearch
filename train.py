""" Defines the Trainer class which handles train/validation/validation_video
"""
import torch
import itertools
import numpy as np
from utils import map
import gc
from utils.utils import AverageMeter, submission_file, Timer, accuracy


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


class Trainer():
    def train(self, loader, model, criterion, optimizer, epoch, args, validate=False):
        timer = Timer()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        metrics = {}

        if validate:
            # switch to evaluate mode
            model.eval()
            criterion.eval()
            iter_size = args.val_size
            setting = 'val epoch'
        else:
            # switch to train mode
            adjust_learning_rate(args.lr, args.lr_decay_rate, optimizer, epoch)
            model.train()
            criterion.train()
            optimizer.zero_grad()
            iter_size = args.train_size
            setting = 'train epoch'

        def part(x):
            return itertools.islice(x, int(len(x)*iter_size))

        for i, (input, target, meta) in enumerate(part(loader)):
            if args.synchronous:
                assert meta['id'][0] == meta['id'][1], "dataset not synced"
                print('all ok with sync')
            gc.collect()
            data_time.update(timer.thetime() - timer.end)

            target = target.long().cuda(async=True)
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target)
            output = model(input_var)
            if type(output) == tuple:
                output, loss, target_var = criterion(*(output + (target_var, meta)))
            else:
                output, loss, target_var = criterion(output, target_var, meta)
            prec1, prec5 = accuracy(output.data, target_var.cpu().data, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

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
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          epoch, i, int(len(loader)*iter_size), len(loader),
                          name=args.name, setting=setting, timer=timer,
                          data_time=data_time, loss=losses, top1=top1, top5=top5))

        if validate:
            metrics.update({'top1val': top1.avg, 'top5val': top5.avg, 'loss_val': losses.avg})
        else:
            metrics.update({'top1train': top1.avg, 'top5train': top5.avg, 'loss_train': losses.avg})
        return metrics

    def validate(self, loader, model, criterion, epoch, args):
        """
            Validate in the same approach as training
        """
        with torch.no_grad():
            return self.train(loader, model, criterion, None, epoch, args, validate=True)

    def validate_video(self, loader, model, criterion, epoch, args):
        """ Run video-level validation on the test set """
        with torch.no_grad():
            timer = Timer()
            outputs = []
            gts = []
            ids = []
            metrics = {}

            # switch to evaluate mode
            model.eval()
            criterion.eval()

            for i, (input, target, meta) in enumerate(loader):
                gc.collect()
                if 'id' in meta:
                    assert meta['id'][0] == meta['id'][1], "val_video not synced"
                    ids.append(meta['id'][0])
                else:
                    assert meta[0]['id'][0] == meta[0]['id'][1], "val_video not synced"
                    ids.append(meta[0]['id'][0])

                target = target.long().cuda(async=True)
                input_var = torch.autograd.Variable(input.cuda(), volatile=True)
                target_var = torch.autograd.Variable(target, volatile=True)

                if args.video_batch_size == -1:
                    output = model(input_var)
                else:
                    output_chunks = []
                    for chunk in input_var.split(args.video_batch_size):
                        output_chunks.append(model(chunk))
                    if type(output_chunks[0]) == tuple:
                        output = tuple(torch.cat(x) for x in zip(*output_chunks))
                    else:
                        output = torch.cat(output_chunks)
                if type(output) == tuple:
                    output, loss, _ = criterion(*(output + (target_var, meta)), synchronous=True)
                else:
                    output, loss, _ = criterion(output, target_var, meta, synchronous=True)

                # store predictions
                #output_video = output.mean(dim=0)
                output_video = output.max(dim=0)[0]
                outputs.append(output_video.data.cpu().numpy())
                if target.dim() == 3:
                    gts.append(target.max(dim=0)[0].max(dim=0)[0])
                else:
                    gts.append(target.max(dim=0)[0])

                timer.tic()
                if i % args.print_freq == 0:
                    print('[{name}] Test2: [{0}/{1}]\t'
                          'Time {timer.val:.3f} ({timer.avg:.3f})'.format(
                              i, len(loader), timer=timer, name=args.name))
            #mAP, _, ap = map.map(np.vstack(outputs), np.vstack(gts))
            mAP, _, ap = map.charades_map(np.vstack(outputs), np.vstack(gts))
            prec1, prec5 = accuracy(torch.Tensor(np.vstack(outputs)), torch.Tensor(np.vstack(gts)), topk=(1, 5))
            print(ap)
            print(' * mAP {:.3f}'.format(mAP))
            print(' * prec1 {:.3f} * prec5 {:.3f}'.format(prec1[0], prec5[0]))
            submission_file(
                ids, outputs, '{}/epoch_{:03d}.txt'.format(args.cache, epoch+1))
            metrics.update({'mAP': mAP, 'videoprec1': prec1[0], 'videoprec5': prec5[0]})
            return metrics
