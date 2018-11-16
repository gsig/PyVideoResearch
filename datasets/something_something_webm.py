""" Dataset loader for the Charades dataset """
from datasets.something_something import SomethingSomething
from datasets.kinetics_mp4 import Kineticsmp4
from datasets.dataset import Dataset
from datasets.utils import cache


class SomethingSomethingwebm(Kineticsmp4, Dataset):
    def __init__(self, args, root, split, labelpath, cachedir,
                 transform=None, target_transform=None, input_size=224, test_gap=10):
        Dataset.__init__(self, test_gap, split)
        self.num_classes = 174
        self.transform = transform
        self.target_transform = target_transform
        self.cls2int = SomethingSomething.parse_something_labels(args.label_file)
        self.labels = SomethingSomething.parse_something_json(labelpath, self.cls2int)
        self.root = root
        self.train_gap = 64
        self.input_size = input_size
        cachename = '{}/{}_{}.pkl'.format(cachedir,
                                          self.__class__.__name__, split)
        self._data = cache(cachename)(self._prepare)(root, self.labels, split)

    def _get_video_path(self, path, vid, label):
        return '{}/{}.webm'.format(path, vid)
