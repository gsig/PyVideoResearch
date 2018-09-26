""" Video loader for the Charades dataset """
import torch
import torchvision.transforms as transforms
import numpy as np
from glob import glob
from datasets.charades import Charades
from datasets.utils import default_loader
import datasets.video_transforms as videotransforms


class CharadesVideo(Charades):
    def __init__(self, *args, **kwargs):
        self.train_gap = 64
        if 'test_gap' not in kwargs:
            kwargs['test_gap'] = 25
        super(CharadesVideo, self).__init__(*args, **kwargs)

    def _prepare(self, path, labels, split):
        datas = []

        for i, (vid, label) in enumerate(labels.iteritems()):
            iddir = path + '/' + vid
            lines = glob(iddir+'/*.jpg')
            n = len(lines)
            if i % 100 == 0:
                print("{} {}".format(i, iddir))
            if n == 0:
                print('empty: {}'.format(iddir))
                continue
            if n <= self.train_gap + 1:
                print('small: {}'.format(iddir))
                continue
            data = {}
            data['base'] = '{}/{}-'.format(iddir, vid)
            data['n'] = n
            data['labels'] = label
            data['id'] = vid
            if split == 'val_video':
                spacing = np.linspace(0, n-self.train_gap-1, self.test_gap)
                for loc in spacing:
                    data['shift'] = loc
                    datas.append(data)
            else:
                datas.append(data)
        return {'datas': datas, 'split': split}

    def __getitem__(self, index, shift=None):
        ims = []
        tars = []
        meta = {}
        fps = 24
        n = self.data['datas'][index]['n']
        if shift is None:
            if hasattr(self.data['datas'][index], 'shift'):
                print('using shift')
                shift = self.data['datas'][index]['shift']
            else:
                shift = np.random.randint(n-self.train_gap-2)

        resize = transforms.Resize(int(256./224*self.input_size))
        spacing = np.arange(shift, shift+self.train_gap)
        for loc in spacing:
            ii = int(np.floor(loc))
            path = '{}{:06d}.jpg'.format(self.data['datas'][index]['base'], ii+1)
            try:
                img = default_loader(path)
            except Exception, e:
                print('failed to load image {}'.format(path))
                print(e)
                raise
            img = resize(img)
            img = transforms.ToTensor()(img)
            img = 2*img - 1
            ims.append(img)
            target = torch.IntTensor(self.num_classes).zero_()
            for x in self.data['datas'][index]['labels']:
                if x['start'] < ii/float(fps) < x['end']:
                    target[self.cls2int(x['class'])] = 1
            tars.append(target)
        meta['id'] = self.data['datas'][index]['id']
        meta['time'] = shift
        img = torch.stack(ims).permute(0, 2, 3, 1).numpy()
        target = torch.stack(tars)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        img = img.transpose([3, 0, 1, 2])
        return img, target, meta

    def __len__(self):
        return len(self.data['datas'])

    @classmethod
    def get(cls, args):
        train_file = args.train_file
        val_file = args.val_file
        train_dataset = cls(
            args, args.data, 'train', train_file, args.cache,
            transform=transforms.Compose([
                videotransforms.RandomCrop(args.input_size),
                videotransforms.RandomHorizontalFlip()
            ]),
            input_size=args.input_size)
        val_dataset = cls(
            args, args.data, 'val', val_file, args.cache,
            transform=transforms.Compose([
                videotransforms.CenterCrop(args.input_size)
            ]),
            input_size=args.input_size)
        valvideo_dataset = cls(
            args, args.data, 'val_video', val_file, args.cache,
            transform=transforms.Compose([
                videotransforms.CenterCrop(args.input_size)
            ]),
            input_size=args.input_size)
        return train_dataset, val_dataset, valvideo_dataset
