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
                 transform=None, target_transform=None, input_size=224, test_gap=50):
        self.num_classes = 157
        self.transform = transform
        self.target_transform = target_transform
        self.labels = self.parse_charades_csv(label_path)
        self.root = root
        self.input_size = input_size
        self.test_gap = test_gap
        cachename = '{}/{}_{}.pkl'.format(cachedir, self.__class__.__name__, split)
        self.data = cache(cachename)(self._prepare)(root, self.labels, split)
        super(Charades, self).__init__(test_gap=test_gap)

    def _prepare(self, path, labels, split):
        fps, gap, test_gap = 24, 4, self.test_gap
        datadir = path
        image_paths, targets, ids, times, ns = [], [], [], [], []

        for i, (vid, label) in enumerate(labels.iteritems()):
            iddir = datadir + '/' + vid
            lines = glob(iddir+'/*.jpg')
            n = len(lines)
            if i % 100 == 0:
                print("{} {}".format(i, iddir))
            if n == 0:
                continue

            if split == 'val_video':
                spacing = np.linspace(0, n-1, test_gap)
            else:
                spacing = range(0, n-1, gap)
            for loc in spacing:
                ii = np.floor(loc)
                target = torch.IntTensor(self.num_classes).zero_()
                for x in label:
                    if split == 'val_video':
                        target[self.cls2int(x['class'])] = 1
                    elif x['start'] < ii/float(fps) < x['end']:
                        target[self.cls2int(x['class'])] = 1
                impath = '{}/{}-{:06d}.jpg'.format(
                    iddir, vid, int(np.floor(loc))+1)
                image_paths.append(impath)
                targets.append(target)
                ids.append(vid)
                times.append(int(np.floor(loc))+1)
                ns.append(n)
        return {'image_paths': image_paths, 'targets': targets, 'ids': ids, 'times': times}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data['image_paths'][index]
        target = self.data['targets'][index]
        meta = {}
        meta['id'] = self.data['ids'][index]
        meta['time'] = self.data['times'][index]
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
    def get(cls, args):
        """ Entry point. Call this function to get all Charades dataloaders """
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_dataset = cls(
            args, args.data, 'train', args.train_file, args.cache,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(args.input_size),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # missing PCA lighting jitter
                normalize,
            ]))
        val_dataset = cls(
            args, args.data, 'val', args.val_file, args.cache,
            transform=transforms.Compose([
                transforms.Resize(int(256./224*args.input_size)),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                normalize,
            ]))
        valvideo_dataset = cls(
            args, args.data, 'val_video', args.val_file, args.cache,
            transform=transforms.Compose([
                transforms.Resize(int(256./224*args.input_size)),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                normalize,
            ]))
        return train_dataset, val_dataset, valvideo_dataset
