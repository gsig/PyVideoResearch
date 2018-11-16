""" Dataset loader for the Charades dataset """
from __future__ import division
import torch
import torchvision.transforms as transforms
import datasets.transforms as arraytransforms
from datasets.charades import Charades
from datasets.utils import flow_loader
import numpy as np
from glob import glob


class Charadesflow(Charades):
    def __init__(self, *args, **kwargs):
        super(Charadesflow, self).__init__(*args, **kwargs)

    def _prepare(self, path, labels, split):
        fps, gap, test_gap = 24, 4, 25
        stack = 10
        datadir = path
        image_paths, targets, ids = [], [], []

        for i, (vid, label) in enumerate(labels.items()):
            iddir = datadir + '/' + vid
            lines = glob(iddir+'/*.jpg')
            n = len(lines)//2
            if i % 100 == 0:
                print("{} {}".format(i, iddir))
            if n == 0:
                continue
            if split == 'val_video':
                target = torch.IntTensor(157).zero_()
                for x in label:
                    target[self.cls2int(x['class'])] = 1
                spacing = np.linspace(0, n-1-stack-1, test_gap)  # fit 10 optical flow pairs
                for loc in spacing:
                    impath = '{}/{}-{:06d}x.jpg'.format(
                        iddir, vid, int(np.floor(loc))+1)
                    image_paths.append(impath)
                    targets.append(target)
                    ids.append(vid)
            else:
                for x in label:
                    for ii in range(0, n-1, gap):
                        if x['start'] < ii/float(fps) < x['end']:
                            if ii > n-1-stack-1:
                                continue  # fit 10 optical flow pairs
                            impath = '{}/{}-{:06d}x.jpg'.format(
                                iddir, vid, ii+1)
                            image_paths.append(impath)
                            targets.append(self.cls2int(x['class']))
                            ids.append(vid)
        return {'image_paths': image_paths, 'targets': targets, 'ids': ids}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data['image_paths'][index]
        base = path[:-5-6]
        framenr = int(path[-5-6:-5])
        assert '{}{:06d}x.jpg'.format(base, framenr) == path
        stack = 10
        img = []
        for i in range(stack):
            x = '{}{:06d}x.jpg'.format(base, framenr+i)
            y = '{}{:06d}y.jpg'.format(base, framenr+i)
            imgx = flow_loader(x)
            imgy = flow_loader(y)
            img.append(imgx)
            img.append(imgy)
        target = self.data['targets'][index]
        meta = {}
        meta['id'] = self.data['ids'][index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, meta

    @classmethod
    def get(cls, args):
        normalize = arraytransforms.Normalize(mean=[0.502], std=[1.0])
        train_dataset = cls(
            args.data, 'train', args.train_file, args.cache,
            transform=transforms.Compose([
                arraytransforms.RandomResizedCrop(224),
                arraytransforms.ToTensor(),
                normalize,
                transforms.Lambda(torch.cat),
            ]))
        val_transforms = transforms.Compose([
            arraytransforms.Resize(256),
            arraytransforms.CenterCrop(224),
            arraytransforms.ToTensor(),
            normalize,
            transforms.Lambda(torch.cat),
        ])
        val_dataset = cls(
            args.data, 'val', args.val_file, args.cache, transform=val_transforms)
        valvideo_dataset = cls(
            args.data, 'val_video', args.val_file, args.cache, transform=val_transforms)
        return train_dataset, val_dataset, valvideo_dataset
