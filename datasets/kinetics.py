""" Dataset loader for the Charades dataset """
import torch
import numpy as np
from glob import glob
from datasets.utils import cache
from datasets.charades import Charades
import csv


def _parse_kinetics_labels(filename):
    labels = {}
    count = 0
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['label'] not in labels:
                labels[row['label']] = count
                count += 1
    return labels


def _parse_kinetics_csv(filename, cls2int):
    labels = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row['youtube_id']
            label = row['label']
            labelnumber = cls2int[label]
            labels[vid] = {'class': labelnumber, 'start': int(row['time_start']), 'end': int(row['time_end'])}
    return labels


class Kinetics(Charades):
    def __init__(self, root, split, labelpath, cachedir, transform=None, target_transform=None, inputsize=224):
        self.num_classes = 400
        self.transform = transform
        self.target_transform = target_transform
        self.cls2int = _parse_kinetics_labels('/nfs.yoda/gsigurds/kinetics400/kinetics_train.csv')
        self.labels = _parse_kinetics_csv(labelpath, self.cls2int)
        self.root = root
        self.testGAP = 25
        cachename = '{}/{}_{}.pkl'.format(cachedir,
                                          self.__class__.__name__, split)
        self.data = cache(cachename)(self.prepare)(root, self.labels, split)

    def prepare(self, path, labels, split):
        GAP, testGAP = 4, self.testGAP
        datadir = path
        image_paths, targets, ids, times = [], [], [], []

        for i, (vid, label) in enumerate(labels.iteritems()):
            iddir = '{}/{}_{:06d}_{:06d}'.format(datadir, vid, label['start'], label['end'])
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
                    impath = '{}/{}_{:06d}_{:06d}_{}.jpg'.format(
                        iddir, vid, label['start'], label['end'], int(np.floor(loc)) + 1)
                    image_paths.append(impath)
                    targets.append(target)
                    ids.append(vid)
                    times.append(int(np.floor(loc)) + 1)
            else:
                for ii in range(0, n - 1, GAP):
                    target = torch.IntTensor(self.num_classes).zero_()
                    target[int(label['class'])] = 1
                    impath = '{}/{}_{:06d}_{:06d}_{}.jpg'.format(
                        iddir, vid, label['start'], label['end'], ii + 1)
                    image_paths.append(impath)
                    targets.append(target)
                    ids.append(vid)
                    times.append(ii)
        return {'image_paths': image_paths, 'targets': targets, 'ids': ids, 'times': times}
