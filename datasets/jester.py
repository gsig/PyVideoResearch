""" Dataset loader for the Charades dataset """
import torch
from datasets.charadesrgb import Charades
from datasets.utils import cache
import numpy as np
from glob import glob


def _parse_jester_labels(filename):
    labels = {}
    with open(filename) as f:
        for i, line in enumerate(f):
            labels[line.strip()] = i
    return labels


def _parse_jester_csv(filename, cls2int):
    labels = {}
    with open(filename) as f:
        for row in f:
            row = row.strip()
            vid, label = row.split(';')
            labelnumber = cls2int[label]
            labels[vid] = {'class': labelnumber}
    return labels


class Jester(Charades):
    def __init__(self, root, split, labelpath, cachedir, transform=None, target_transform=None):
        self.num_classes = 27
        self.transform = transform
        self.target_transform = target_transform
        self.cls2int = _parse_jester_labels('/nfs.yoda/gsigurds/jester/jester-v1-labels.csv')
        self.labels = _parse_jester_csv(labelpath, self.cls2int)
        self.root = root
        self.testGAP = 50
        cachename = '{}/{}_{}.pkl'.format(cachedir,
                                          self.__class__.__name__, split)
        self.data = cache(cachename)(self.prepare)(root, self.labels, split)

    def prepare(self, path, labels, split):
        FPS, GAP, testGAP = 24, 4, self.testGAP
        datadir = path
        image_paths, targets, ids, times = [], [], [], []

        for i, (vid, label) in enumerate(labels.iteritems()):
            iddir = datadir + '/' + vid
            lines = glob(iddir + '/*.jpg')
            n = len(lines)
            if i % 1000 == 0:
                print("{} {}".format(i, iddir))
            if n == 0:
                continue
            if split == 'val_video':
                target = torch.IntTensor(self.num_classes).zero_()
                target[int(label['class'])] = 1
                spacing = np.linspace(0, n - 1, testGAP)
                for loc in spacing:
                    impath = '{}/{:05d}.jpg'.format(
                        iddir, int(np.floor(loc)) + 1)
                    image_paths.append(impath)
                    targets.append(target)
                    ids.append(vid)
                    times.append(int(np.floor(loc)) + 1)
            else:
                for ii in range(0, n - 1, GAP):
                    target = torch.IntTensor(self.num_classes).zero_()
                    target[int(label['class'])] = 1
                    impath = '{}/{:05d}.jpg'.format(
                        iddir, ii + 1)
                    image_paths.append(impath)
                    targets.append(target)
                    ids.append(vid)
                    times.append(ii)
        return {'image_paths': image_paths, 'targets': targets, 'ids': ids, 'times': times}
