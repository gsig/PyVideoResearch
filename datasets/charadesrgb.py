""" Dataset loader for the Charades dataset """
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import random
import numpy as np
from glob import glob
import csv
import cPickle as pickle
import os


def parse_charades_csv(filename):
    labels = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row['id']
            actions = row['actions']
            if actions == '':
                actions = []
            else:
                actions = [a.split(' ') for a in actions.split(';')]
                actions = [{'class': x, 'start': float(
                    y), 'end': float(z)} for x, y, z in actions]
            labels[vid] = actions
    return labels


def cls2int(x):
    return int(x[1:])


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def cache(cachefile, bust=False):
    """ Creates a decorator that caches the result to cachefile """
    def cachedecorator(fn):
        def newf(*args, **kwargs):
            print('cachefile {}'.format(cachefile))
            if os.path.exists(cachefile) and not bust:
                with open(cachefile, 'rb') as f:
                    print("Loading cached result from '%s'" % cachefile)
                    return pickle.load(f)
            res = fn(*args, **kwargs)
            with open(cachefile, 'wb') as f:
                print("Saving result to cache '%s'" % cachefile)
                pickle.dump(res, f)
            return res
        return newf
    return cachedecorator


class Charades(data.Dataset):
    def __init__(self, root, split, labelpath, cachedir, bust, transform=None, target_transform=None):
        self.num_classes = 157
        self.transform = transform
        self.target_transform = target_transform
        self.labels = parse_charades_csv(labelpath)
        self.root = root
        cachename = '{}/{}_{}.pkl'.format(cachedir,
                                          self.__class__.__name__, split)
        self.data = cache(cachename, bust)(self.prepare)(root, self.labels, split)
        print('{} samples loaded'.format(self.__len__()))

    def prepare(self, path, labels, split):
        FPS, GAP, testGAP = 24, 4, 25
        datadir = path
        image_paths, targets, ids = [], [], []

        for i, (vid, label) in enumerate(labels.iteritems()):
            iddir = datadir + '/' + vid
            lines = glob(iddir + '/*.jpg')
            n = len(lines)
            if i % 100 == 0:
                print("{} {}".format(i, iddir))
            if n == 0:
                continue
            if split == 'val_video':
                target = torch.IntTensor(157).zero_()
                for x in label:
                    target[cls2int(x['class'])] = 1
                spacing = np.linspace(0, n - 1, testGAP)
                for loc in spacing:
                    impath = '{}/{}-{:06d}.jpg'.format(
                        iddir, vid, int(np.floor(loc)) + 1)
                    image_paths.append(impath)
                    targets.append(target)
                    ids.append(vid)
            else:
                for x in label:
                    for ii in range(0, n - 1, GAP):
                        if x['start'] < ii / float(FPS) < x['end']:
                            impath = '{}/{}-{:06d}.jpg'.format(
                                iddir, vid, ii + 1)
                            image_paths.append(impath)
                            targets.append(cls2int(x['class']))
                            ids.append(vid)
        return {'image_paths': image_paths, 'targets': targets, 'ids': ids}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data['image_paths'][index]
        target = self.data['targets'][index]
        meta = {}
        meta['id'] = self.data['ids'][index]
        try:
            img = default_loader(path)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target, meta
        except Exception:
            print('failed loading item: {}'.format(path))
            print('fetching another random item instead')
            return self[random.randrange(len(self))]

    def __len__(self):
        return len(self.data['image_paths'])

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    @classmethod
    def get(cls, args, scale=(0.08, 1.0)):
        """ Entry point. Call this function to get all Charades dataloaders """
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_file = args.train_file
        val_file = args.val_file
        train_dataset = cls(
            args.data, 'train', train_file, args.cache, args.cache_buster,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(args.inputsize, scale),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # missing PCA lighting jitter
                normalize,
            ]))
        val_dataset = cls(
            args.data, 'val', val_file, args.cache, args.cache_buster,
            transform=transforms.Compose([
                transforms.Resize(int(256. / 224 * args.inputsize)),
                transforms.CenterCrop(args.inputsize),
                transforms.ToTensor(),
                normalize,
            ]))
        valvideo_dataset = cls(
            args.data, 'val_video', val_file, args.cache, args.cache_buster,
            transform=transforms.Compose([
                transforms.Resize(int(256. / 224 * args.inputsize)),
                transforms.CenterCrop(args.inputsize),
                transforms.ToTensor(),
                normalize,
            ]))
        return train_dataset, val_dataset, valvideo_dataset
