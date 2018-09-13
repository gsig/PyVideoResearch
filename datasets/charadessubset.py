""" Dataset loader for the Charades dataset """
from datasets.charades import Charades, cache, _parse_charades_csv


class CharadesSubset(Charades):
    def __init__(self, root, split, labelpath, cachedir, transform=None, target_transform=None, inputsize=224):
        self.num_classes = 157
        self.transform = transform
        self.target_transform = target_transform
        self.labels = _parse_charades_csv(labelpath)
        self.labels = self.labels[:100]
        self.root = root
        if not hasattr(self, 'testGAP'):
            self.testGAP = 50
        cachename = '{}/{}_{}.pkl'.format(cachedir,
                                          self.__class__.__name__, split)
        self.data = cache(cachename)(self.prepare)(root, self.labels, split)
