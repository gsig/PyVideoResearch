import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self, test_gap=None):
        if 'test_gap' is None:
            raise NotImplementedError()
        self.test_gap = test_gap

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
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
        """ Entry point. Call this function to get all dataloaders """
        raise NotImplementedError()
