""" Dataset loader for the Charades dataset """
import torch
from datasets.dataset import Dataset
import random


class MockDataset1(Dataset):
    def __init__(self, split):
        super(Dataset, self).__init__(test_gap, split)
        self.num_classes = 5
        self._data = self.prepare(split)

    def prepare(self, split):
        targets = []
        ids = []
        times = []
        inputs = []
        for j in range(100):
            target = 0 if random.random() < .5 else 1
            for i in range(self.test_gap):
                if split == 'val_video':
                    tmp = torch.zeros(5)
                    tmp[target] = 1
                    targets.append(tmp)
                else:
                    targets.append(target)
                inp = target
                if random.random() < .25:
                    inp = 1 - inp
                inputs.append(torch.zeros(100) + inp)
                ids.append(split + str(j))
                times.append(i)
        return {'inputs': inputs, 'targets': targets, 'ids': ids, 'times': times}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        inp = self.data['inputs'][index]
        target = self.data['targets'][index]
        meta = {}
        meta['id'] = self.data['ids'][index]
        meta['time'] = self.data['times'][index]
        return inp, target, meta

    def __len__(self):
        return len(self.data['inputs'])

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        return fmt_str

    @classmethod
    def get(cls, args):
        train_dataset = Testdata1('train')
        val_dataset = Testdata1('val')
        valvideo_dataset = Testdata1('val_video')
        return train_dataset, val_dataset, valvideo_dataset
