""" Dataset loader for the Charades dataset """
from glob import glob
import json
from datasets.charades import Charades
from datasets.dataset import Dataset
from datasets.utils import cache
import numpy as np
import torch


class SomethingSomething(Charades, Dataset):
    def __init__(self, args, root, split, labelpath, cachedir, transform=None, target_transform=None, test_gap=50):
        Dataset.__init__(self, test_gap, split)
        self.num_classes = 174
        self.transform = transform
        self.target_transform = target_transform
        self.cls2int = self.parse_something_labels(args.label_file)
        self.labels = self.parse_something_json(labelpath, self.cls2int)
        self.root = root
        cachename = '{}/{}_{}.pkl'.format(cachedir, self.__class__.__name__, split)
        self._data = cache(cachename)(self._prepare)(root, self.labels, split)

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
                ii = np.floor(loc)
                impath = '{}/{}-{:06d}.jpg'.format(iddir, vid, ii + 1)
                image_paths.append(impath)
                targets.append(target)
                ids.append(vid)
                times.append(ii + 1)
        return {'image_paths': image_paths, 'targets': targets, 'ids': ids, 'times': times}

    @staticmethod
    def parse_something_labels(filename):
        with open(filename) as f:
            labels = json.load(f)
        return labels

    @staticmethod
    def parse_something_json(filename, cls2int):
        labels = {}
        with open(filename) as f:
            data = json.load(f)
        for row in data:
            vid = row['id']
            label = row['template'].replace('[', '').replace(']', '')
            labelnumber = int(cls2int[label])
            labels[vid] = {'class': labelnumber, 'vid': vid}
        return labels
