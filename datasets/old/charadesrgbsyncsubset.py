""" Dataset loader for the Charades dataset """
import torch
import torchvision.transforms as transforms
import numpy as np
from glob import glob
from charadesrgb import Charades, cls2int


class CharadesSyncSub(Charades):
    def __init__(self, *args, **kwargs):
        self.testGAP = 64
        super(CharadesSyncSub, self).__init__(*args, **kwargs)

    def prepare(self, path, labels, split):
        FPS, GAP, testGAP = 24, 4, self.testGAP
        datadir = path
        image_paths, targets, ids, times = [], [], [], []

        for i, (vid, label) in enumerate(labels.iteritems()):
            if i>100: continue
            iddir = datadir + '/' + vid
            lines = glob(iddir+'/*.jpg')
            n = len(lines)
            if i % 100 == 0:
                print("{} {}".format(i, iddir))
            if n == 0:
                print('empty: {}'.format(iddir))
                continue
            if split == 'val_video':
                target = torch.IntTensor(self.num_classes).zero_()
                for x in label:
                    target[cls2int(x['class'])] = 1
                spacing = np.linspace(0, n-1, testGAP)
                for loc in spacing:
                    impath = '{}/{}-{:06d}.jpg'.format(
                        iddir, vid, int(np.floor(loc))+1)
                    image_paths.append(impath)
                    targets.append(target)
                    ids.append(vid)
                    times.append(int(np.floor(loc))+1)
            else:
                spacing = np.linspace(0, n-1, testGAP)
                for loc in spacing:
                    ii = int(np.floor(loc))
                    target = torch.IntTensor(self.num_classes).zero_()
                    for x in label:
                        if x['start'] < ii/float(FPS) < x['end']:
                            target[cls2int(x['class'])] = 1
                    impath = '{}/{}-{:06d}.jpg'.format(
                        iddir, vid, ii+1)
                    image_paths.append(impath)
                    targets.append(target)
                    ids.append(vid)
                    times.append(ii+1)
        return {'image_paths': image_paths, 'targets': targets, 'ids': ids, 'times': times}


def get(args):
    """ Entry point. Call this function to get all Charades dataloaders """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_file = args.train_file
    val_file = args.val_file
    train_dataset = CharadesSyncSub(
        args.data, 'train', train_file, args.cache,
        transform=transforms.Compose([
            #transforms.RandomResizedCrop(args.inputsize),
            #transforms.ColorJitter(
            #    brightness=0.4, contrast=0.4, saturation=0.4),
            #transforms.RandomHorizontalFlip(), # TODO augmentation
            transforms.Resize(int(256./224*args.inputsize)),
            transforms.CenterCrop(args.inputsize),
            transforms.ToTensor(),  # missing PCA lighting jitter
            normalize,
        ]))
    val_dataset = CharadesSyncSub(
        args.data, 'val', val_file, args.cache,
        transform=transforms.Compose([
            transforms.Resize(int(256./224*args.inputsize)),
            transforms.CenterCrop(args.inputsize),
            transforms.ToTensor(),
            normalize,
        ]))
    valvideo_dataset = CharadesSyncSub(
        args.data, 'val_video', val_file, args.cache,
        transform=transforms.Compose([
            transforms.Resize(int(256./224*args.inputsize)),
            transforms.CenterCrop(args.inputsize),
            transforms.ToTensor(),
            normalize,
        ]))
    return train_dataset, val_dataset, valvideo_dataset
