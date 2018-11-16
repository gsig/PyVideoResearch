""" Dataset loader for the Charades dataset """
from datasets.charades import Charades
from datasets.utils import cache


class CharadesSubset(Charades):
    def __init__(self, args, root, split, labelpath, cachedir,
                 transform=None, target_transform=None, input_size=224, test_gap=50):
        self.num_classes = 157
        self.transform = transform
        self.target_transform = target_transform
        self.labels = self.parse_charades_csv(labelpath)
        self.labels = self.labels[:100]
        self.root = root
        self.test_gap = test_gap
        cachename = '{}/{}_{}.pkl'.format(cachedir, self.__class__.__name__, split)
        self._data = cache(cachename)(self._prepare)(root, self.labels, split)
