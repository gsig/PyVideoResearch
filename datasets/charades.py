""" Dataset loader for the Charades dataset """
import torch
import torchvision.transforms as transforms
from datasets.dataset import Dataset
from datasets.utils import default_loader, cache
import numpy as np
from glob import glob
import csv


class Charades(Dataset):
    def __init__(self, args, root, split, label_path, cachedir,
                 transform=None, target_transform=None, input_size=224, test_gap=50, train_gap=4, fps=24):
        super(Charades, self).__init__(test_gap, split)
        self.num_classes = 157
        self.transform = transform
        self.target_transform = target_transform
        self.labels = self.parse_charades_csv(label_path)
        self.root = root
        self.input_size = input_size
        self.fps = fps
        self.train_gap = train_gap
        cachename = '{}/{}_{}.pkl'.format(cachedir, self.__class__.__name__, split)
        self._data = cache(cachename)(self._prepare)(root, self.labels, split)

    @property
    def data(self):
        return self._data

    def _prepare(self, path, labels, split):
        fps, gap = self.fps, self.train_gap
        datadir = path
        image_paths, ids, times, ns, alllabels = [], [], [], [], []

        for i, (vid, label) in enumerate(labels.items()):
            iddir = datadir + '/' + vid
            lines = glob(iddir+'/*.jpg')
            n = len(lines)
            import pdb
            pdb.set_trace()
            if i % 100 == 0:
                print("{} {}".format(i, iddir))
            if n == 0:
                continue
            if split == 'val_video':
                spacing = [0]
            else:
                spacing = range(0, n-1, gap)
            for loc in spacing:
                ii = np.floor(loc)
                impath = '{}/{}-{:06d}.jpg'.format(
                    iddir, vid, int(np.floor(loc))+1)
                image_paths.append(impath)
                ids.append(vid)
                times.append(int(np.floor(loc))+1)
                ns.append(n)
                alllabels.append(label)
        return {'image_paths': image_paths, 'ids': ids, 'times': times, 'ns': ns, 'labels': alllabels, 'split': split}

    def get_item(self, index, shift=None):
        meta = {}
        if shift is None:
            path = self.data['image_paths'][index]
            shift = meta['time'] = self.data['times'][index]
        else:
            n = self.data['ns'][index]
            shift = meta['time'] = int(shift * (n-1))
            base = self.data['image_paths'][index][:-10]
            path = '{}{:06d}.jpg'.format(base, shift+1)
        target = torch.IntTensor(self.num_classes).zero_()
        for x in self.data['labels'][index]:
            if x['start'] < shift/float(self.fps) < x['end']:
                target[self.cls2int(x['class'])] = 1
        meta['id'] = self.data['ids'][index]
        img = default_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, meta

    @staticmethod
    def parse_charades_csv(filename):
        labels = {}
        with open(filename) as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = row['id']
                actions = row['actions']
                if actions == '':
                    actions = []
                else:
                    actions = [a.split(' ') for a in actions.split(';')]
                    actions = [{'class': x, 'start': float(
                        y), 'end': float(z)} for x, y, z in actions]
                labels[vid] = actions
        return labels

    @staticmethod
    def cls2int(x):
        return int(x[1:])

    def __len__(self):
        return len(self.data['image_paths'])

    @classmethod
    def get(cls, args, scale=(0.08, 1.0), splits=('train', 'val', 'val_video')):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if 'train' in splits:
            train_dataset = cls(
                args, args.data, 'train', args.train_file, args.cache,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(args.input_size, scale),
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),  # missing PCA lighting jitter
                    normalize,
                ]))
        else:
            train_dataset = None
        if 'val' in splits:
            val_dataset = cls(
                args, args.data, 'val', args.val_file, args.cache,
                transform=transforms.Compose([
                    transforms.Resize(int(256./224*args.input_size)),
                    transforms.CenterCrop(args.input_size),
                    transforms.ToTensor(),
                    normalize,
                ]))
        else:
            val_dataset = None
        if 'val_video' in splits:
            valvideo_dataset = cls(
                args, args.data, 'val_video', args.val_file, args.cache,
                transform=transforms.Compose([
                    transforms.Resize(int(256./224*args.input_size)),
                    transforms.CenterCrop(args.input_size),
                    transforms.ToTensor(),
                    normalize,
                ]))
        else:
            valvideo_dataset = None
        return train_dataset, val_dataset, valvideo_dataset
