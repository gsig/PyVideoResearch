""" Dataset loader for the ActivityNet dataset """
import json
from datasets.dataset_jpg import DatasetJPG
from glob import glob
import numpy as np


class ActivityNet2(DatasetJPG):
    def __init__(self, *args, **kwargs):
        if 'train_gap' not in kwargs:
            kwargs['train_gap'] = 1
        if 'fps' not in kwargs:
            kwargs['fps'] = 4
        if 'num_classes' not in kwargs:
            kwargs['num_classes'] = 200
        super(ActivityNet2, self).__init__(*args, **kwargs)

    def get_jpg_path(self, base, vid, i):
        return '{}/{}_{}.{}'.format(base, vid, i+1, self.ext)

    def _prepare(self, path, labels, split):
        datadir = path
        image_paths, datas = [], []

        for i, (vid, label) in enumerate(labels.items()):
            iddir = self.get_video_basedir(datadir, vid)
            lines = glob(iddir+'/*.{}'.format(self.ext))
            n = len(lines)
            if i % 100 == 0:
                print("{} {}".format(i, iddir))
            if n == 0:
                continue
            if split == 'val_video':
                ii = 0
                impath = self.get_jpg_path(iddir, vid, ii)
                image_paths.append(impath)  # legacy
                data = {'base': iddir,
                        'labels': label,
                        'id': vid,
                        'time': ii,
                        'n': n}
                datas.append(data)
            else:
                spacing = range(0, n-1, self.train_gap)
                for loc in spacing:
                    for x in label:
                        if x['start'] < loc/float(self.fps) < x['end']:
                            ii = int(np.floor(loc))
                            impath = self.get_jpg_path(iddir, vid, ii)
                            image_paths.append(impath)  # legacy
                            data = {'base': iddir,
                                    'labels': label,
                                    'id': vid,
                                    'time': ii,
                                    'n': n}
                            datas.append(data)

        return {'image_paths': image_paths,
                'datas': datas,
                'split': split}

    @staticmethod
    def get_label_map(filename):
        cls2int = {}
        clsint = 0
        with open(filename) as f:
            data = json.load(f)
        for _, row in data['database'].items():
            if 'training' != row['subset']:
                continue
            for ann in row['annotations']:
                label = ann['label']
                if label not in cls2int:
                    cls2int[label] = clsint
                    clsint += 1
        return cls2int

    @staticmethod
    def get_labels(filename, split, cls2int):
        if split == 'val' or split == 'val_video':
            split = 'validation'
        elif split == 'train':
            split = 'training'
        else:
            assert False, "invalid split"
        labels = {}
        with open(filename) as f:
            data = json.load(f)
        for vid, row in data['database'].items():
            if split != row['subset']:
                continue
            actions = []
            for ann in row['annotations']:
                label = ann['label']
                start = float(ann['segment'][0])
                end = float(ann['segment'][1])
                actions.append({'class': label,
                                'vid': vid,
                                'start': start,
                                'end': end})
            labels[vid] = actions
        return labels
