""" Dataset loader for the Charades dataset """
import torch
import numpy as np
from glob import glob
from datasets.utils import cache
from datasets.charades import Charades
from datasets.dataset import Dataset
import csv


class Kinetics(Charades, Dataset):
    def __init__(self, args, root, split, label_path, cachedir,
                 transform=None, target_transform=None, input_size=224, test_gap=25, train_gap=4):
        Dataset.__init__(self, test_gap, split)
        self.num_classes = 400
        self.transform = transform
        self.target_transform = target_transform
        self.cls2int = self.parse_kinetics_labels(args.train_file)
        self.labels = self.parse_kinetics_csv(label_path, self.cls2int)
        self.root = root
        self.train_gap = train_gap
        self.input_size = input_size
        cachename = '{}/{}_{}.pkl'.format(cachedir, self.__class__.__name__, split)
        self._data = cache(cachename)(self._prepare)(root, self.labels, split)

    def _prepare(self, path, labels, split):
        gap, test_gap = self.train_gap, self.test_gap
        datadir = path
        image_paths, targets, ids, times = [], [], [], []

        for i, (vid, label) in enumerate(labels.items()):
            iddir = '{}/{}_{:06d}_{:06d}'.format(datadir, vid, label['start'], label['end'])
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
                    ii = int(np.floor(loc))
                    impath = '{}/{}_{:06d}_{:06d}_{}.jpg'.format(
                        iddir, vid, label['start'], label['end'], ii + 1)
                    image_paths.append(impath)
                    targets.append(target)
                    ids.append(vid)
                    times.append(ii + 1)
        return {'image_paths': image_paths, 'targets': targets, 'ids': ids, 'times': times}

    @staticmethod
    def parse_kinetics_labels(filename):
        labels = {}
        count = 0
        with open(filename) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['label'] not in labels:
                    labels[row['label']] = count
                    count += 1
        return labels

    @staticmethod
    def parse_kinetics_csv(filename, cls2int):
        labels = {}
        with open(filename) as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = row['youtube_id']
                label = row['label']
                labelnumber = cls2int[label]
                labels[vid] = {
                    'vid': vid,
                    'class': labelnumber,
                    'start': int(row['time_start']),
                    'end': int(row['time_end'])}
        return labels
