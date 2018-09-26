""" Dataset loader for the Charades dataset """
import torch
import torchvision.transforms as transforms
from datasets.kinetics import Kinetics
from PIL import Image
import numpy as np
import datasets.video_transforms as videotransforms
import os
from datasets.utils import ffmpeg_video_loader as video_loader


class Kineticsmp4(Kinetics):
    def __init__(self, *args, **kwargs):
        self.train_gap = 64
        if 'test_gap' not in kwargs:
            kwargs['test_gap'] = 10
        self.input_size = kwargs['input_size']
        super(Kineticsmp4, self).__init__(*args, **kwargs)

    def _prepare(self, path, labels, split):
        datadir = path
        int2cls = dict([(y, x) for x, y in self.cls2int.items()])
        datas = []

        for i, (vid, label) in enumerate(labels.iteritems()):
            name = int2cls[label['class']].replace(' ', '_')
            iddir = '{}/{}/{}_{:06d}_{:06d}.mp4'.format(
                datadir, name, vid, label['start'], label['end'])
            n = int(float(label['end']) - float(label['start']))
            if i % 1000 == 0:
                print("{} {}".format(i, iddir))
            if not os.path.isfile(iddir):
                print('empty: {}'.format(iddir))
                continue
            data = {}
            data['base'] = iddir
            data['n'] = n
            data['labels'] = label
            data['id'] = vid
            if split == 'val_video':
                spacing = np.linspace(0, 1.0, self.test_gap)
                for loc in spacing:
                    data['shift'] = loc
                    datas.append(data)
            else:
                datas.append(data)
        return {'datas': datas, 'split': split}

    def __getitem__(self, index):
        ims = []
        tars = []
        meta = {}
        path = self.data['datas'][index]['base']
        try:
            video, fps = video_loader(path)
        except (TypeError, Exception) as e:
            print('failed to load video {}'.format(path))
            print(e)
            #return self[np.random.randint(len(self))]
            return self[index+1]
        n = video.shape[0]
        if hasattr(self.data['datas'][index], 'shift'):
            print('using shift')
            shift = self.data['datas'][index]['shift']
            shift = int(shift * (n-self.train_gap-2))
        else:
            if n <= self.train_gap+2:
                shift = 0
            else:
                shift = np.random.randint(n-self.train_gap-2)

        resize = transforms.Resize(int(256./224*self.input_size))
        spacing = np.arange(shift, shift+self.train_gap)
        for loc in spacing:
            if loc >= len(video):
                img = video[-1]
            else:
                img = video[loc]
            img = resize(Image.fromarray(img))
            img = transforms.ToTensor()(img)
            img = 2*img - 1
            ims.append(img)
            target = torch.IntTensor(self.num_classes).zero_()
            target[self.data['datas'][index]['labels']['class']] = 1
            tars.append(target)
        meta['id'] = self.data['datas'][index]['id']
        meta['time'] = shift
        img = torch.stack(ims).permute(0, 2, 3, 1).numpy()  # n, h, w, c
        target = torch.stack(tars)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        img = img.transpose([3, 0, 1, 2])  # c, n, h, w
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
            args, args.valdata, 'val', val_file, args.cache,
            transform=transforms.Compose([
                videotransforms.CenterCrop(args.input_size)
            ]),
            input_size=args.input_size)
        valvideo_dataset = cls(
            args, args.valdata, 'val_video', val_file, args.cache,
            transform=transforms.Compose([
                videotransforms.CenterCrop(args.input_size)
            ]),
            input_size=args.input_size)
        return train_dataset, val_dataset, valvideo_dataset
