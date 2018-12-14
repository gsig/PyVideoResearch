import torch.utils.data as data
import numpy as np


class Dataset(data.Dataset):
    def __init__(self, test_gap, split):
        self._test_gap = test_gap
        self._split = split

    @property
    def test_gap(self):
        """
        Number of locations across the video to evaluate the model
        """
        return self._test_gap

    @property
    def split(self):
        """
        String with the name of the dataset split
        By default this is either 'train', 'val', 'val_video'
        """
        return self._split

    @property
    def data(self):
        """
        Container for dataset info
        """
        raise NotImplementedError

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            if self.split == 'val video'
                list of tuples: (image, target)
            else:
                tuple: (image, target) target is label tensor
        """
        try:
            if self.split == 'val_video':
                print('preparing video across {} locations'.format(self.test_gap))
                return [self.get_item(index, shift=t)
                        for t in np.linspace(0, 1.0, self.test_gap)]
            else:
                return self.get_item(index, shift=None)
        except Exception as e:
            print('error getting item {}, moving on to next item'.format(index))
            print(e)
            return self.__getitem__((index + 1) % len(self))

    def get_item(self, index, shift=None, video=None):
        """
        Args:
            index (int): Index
            shift (float): [0,1] % location in video
            video: optional video cache
        Returns:
            tuple: (image, target) target is label array
        """
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        if hasattr(self, 'split'):
            fmt_str += '    Split: {}\n'.format(self._split)
        if hasattr(self, 'root'):
            fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        if hasattr(self, 'transform'):
            fmt_str += '{0}{1}\n'.format(
                tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        if hasattr(self, 'target_transform'):
            fmt_str += '{0}{1}'.format(
                tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    @classmethod
    def get(cls, self):
        """ Entry point. Call this function to get all dataloaders
            Returns:
                train_dataset, val_dataset, valvideo_dataset
        """
        raise NotImplementedError()
