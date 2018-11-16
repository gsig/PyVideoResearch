""" Dataset loader for the ActivityNet dataset """
import json
from datasets.dataset_jpg import DatasetJPG


class ActivityNet(DatasetJPG):
    def __init__(self, *args, **kwargs):
        if 'train_gap' not in kwargs:
            kwargs['train_gap'] = 4
        if 'fps' not in kwargs:
            kwargs['fps'] = 4
        if 'num_classes' not in kwargs:
            kwargs['num_classes'] = 200
        super(ActivityNet, self).__init__(*args, **kwargs)

    def get_jpg_path(self, base, vid, i):
        return '{}/{}_{}.{}'.format(base, vid, i+1, self.ext)

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
