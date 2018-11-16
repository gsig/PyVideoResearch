""" Dataset loader for the Charades dataset """
import torch
from datasets.charades import Charades
from datasets.utils import cache
import numpy as np
from glob import glob


class Jester(Charades):
    def __init__(self, args, root, split, labelpath, cachedir, transform=None, target_transform=None, test_gap=50):
        self.num_classes = 27
        self.transform = transform
        self.target_transform = target_transform
        self.cls2int = self.parse_jester_labels(args.label_file)
        self.labels = self.parse_jester_csv(labelpath, self.cls2int)
        self.root = root
        self.test_gap = test_gap
        cachename = '{}/{}_{}.pkl'.format(cachedir, self.__class__.__name__, split)
        self.data = cache(cachename)(self._prepare)(root, self.labels, split)

    def _prepare(self, path, labels, split):
        gap, test_gap = 4, self.test_gap
        datadir = path
        image_paths, targets, ids, times = [], [], [], []

        for i, (vid, label) in enumerate(labels.items()):
            iddir = datadir + '/' + vid
            lines = glob(iddir + '/*.jpg')
            n = len(lines)
            if i % 1000 == 0:
                print("{} {}".format(i, iddir))
            if n == 0:
                continue
            if split == 'val_video':
                spacing = np.linspace(0, n - 1, test_gap)
            else:
                spacing = range(0, n - 1, gap)
            target = torch.IntTensor(self.num_classes).zero_()
            target[int(label['class'])] = 1
            for loc in spacing:
                impath = '{}/{:05d}.jpg'.format(
                    iddir, int(np.floor(loc)) + 1)
                image_paths.append(impath)
                targets.append(target)
                ids.append(vid)
                times.append(int(np.floor(loc)) + 1)
        return {'image_paths': image_paths, 'targets': targets, 'ids': ids, 'times': times}

    @staticmethod
    def parse_jester_labels(filename):
        labels = {}
        with open(filename) as f:
            for i, line in enumerate(f):
                labels[line.strip()] = i
        return labels

    @staticmethod
    def parse_jester_csv(filename, cls2int):
        labels = {}
        with open(filename) as f:
            for row in f:
                row = row.strip()
                vid, label = row.split(';')
                labelnumber = cls2int[label]
                labels[vid] = {'class': labelnumber}
        return labels
