""" Dataset loader for the Charades dataset """
import torch
import torchvision.transforms as transforms
from datasets.dataset import Dataset
from datasets.utils import default_loader, cache
import numpy as np
from glob import glob


class DatasetJPG(Dataset):
    def __init__(self, args, root, split, label_path, cachedir,
                 transform=None, target_transform=None,
                 input_size=224, test_gap=50, train_gap=4,
                 fps=24, num_classes=157, ext='jpg'):
        super(DatasetJPG, self).__init__(test_gap, split)
        self.num_classes = num_classes
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.input_size = input_size
        self.fps = fps
        self.train_gap = train_gap
        self.ext = ext
        self.cls2int = self.get_label_map(args.label_file)
        self.labels = self.get_labels(label_path, split, self.cls2int)
        cachename = '{}/{}_{}.pkl'.format(cachedir, self.__class__.__name__, split)
        self._data = cache(cachename)(self._prepare)(root, self.labels, split)

    @staticmethod
    def get_labels(label_path, split, cls2int):
        raise NotImplementedError

    @staticmethod
    def get_label_map(label_file):
        raise NotImplementedError

    def get_video_basedir(self, datadir, vid):
        return datadir + '/' + vid

    def get_jpg_path(self, base, vid, i):
        return '{}/{}-{:06d}.{}'.format(base, vid, i+1, self.ext)

    @property
    def data(self):
        return self._data

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
                spacing = [0]
            else:
                spacing = range(0, n-1, self.train_gap)
            for loc in spacing:
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

    def get_item(self, index, shift=None):
        meta = {}
        meta['do_not_collate'] = True
        datas = self.data['datas'][index]
        if shift is None:
            path = self.data['image_paths'][index]
            meta['time'] = shift = datas['time']
        else:
            n = datas['n']
            shift = int(shift * (n-1))
            base = datas['base']
            path = self.get_jpg_path(base, datas['id'], shift)
            meta['time'] = shift

        target = torch.IntTensor(self.num_classes).zero_()
        for x in datas['labels']:
            if x['start'] < shift/float(self.fps) < x['end']:
                target[self.cls2int[x['class']]] = 1

        meta['id'] = datas['id']
        img = default_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, meta

    def __len__(self):
        return len(self.data['datas'])

    @classmethod
    def get(cls, args):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_dataset = cls(
            args, args.data, 'train', args.train_file, args.cache,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(args.input_size),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # missing PCA lighting jitter
                normalize,
            ]))
        val_dataset = cls(
            args, args.data, 'val', args.val_file, args.cache,
            transform=transforms.Compose([
                transforms.Resize(int(256./224*args.input_size)),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                normalize,
            ]))
        valvideo_dataset = cls(
            args, args.data, 'val_video', args.val_file, args.cache,
            transform=transforms.Compose([
                transforms.Resize(int(256./224*args.input_size)),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                normalize,
            ]))
        return train_dataset, val_dataset, valvideo_dataset
