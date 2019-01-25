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
        if 'train_gap' not in kwargs:
            kwargs['train_gap'] = 64
        if 'test_gap' not in kwargs:
            kwargs['test_gap'] = 10
        super(Kineticsmp4, self).__init__(*args, **kwargs)

    def _get_video_path(self, path, vid, label):
        if not hasattr(self, 'int2cls'):
            self.int2cls = dict([(y, x) for x, y in self.cls2int.items()])
        name = self.int2cls[label['class']].replace(' ', '_')
        iddir = '{}/{}/{}_{:06d}_{:06d}.mp4'.format(
            path, name, vid, label['start'], label['end'])
        return iddir

    def _prepare(self, path, labels, split):
        datas = []
        for i, (k, label) in enumerate(labels.items()):
            vid = label['vid']
            iddir = self._get_video_path(path, vid, label)
            if i % 1000 == 0:
                print("{} {}".format(i, iddir))
            if not os.path.isfile(iddir):
                print('empty: {}'.format(iddir))
                continue
            data = {}
            data['base'] = iddir
            data['labels'] = label
            data['id'] = vid
            datas.append(data)
        return {'datas': datas, 'split': split}

    def _process_stack(self, video, shift, data):
        ims, tars, meta = [], [], {}
        meta['do_not_collate'] = True
        if self.split == 'train' and np.random.random() > 0.5:
            resize = transforms.Resize(int(320./224*self.input_size))
        else:
            resize = transforms.Resize(int(256./224*self.input_size))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        spacing = np.arange(shift, shift+self.train_gap)
        for loc in spacing:
            img = video[loc] if loc < len(video) else video[-1]
            img = resize(Image.fromarray(img))
            img = transforms.ToTensor()(img)
            #img = 2*img - 1
            img = normalize(img)
            ims.append(img)
            target = torch.IntTensor(self.num_classes).zero_()
            target[data['labels']['class']] = 1
            tars.append(target)
        meta['id'] = data['id']
        meta['time'] = shift
        img = torch.stack(ims).permute(0, 2, 3, 1).numpy()  # n, h, w, c
        target = torch.stack(tars)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, meta

    def __getitem__(self, index):
        path = self.data['datas'][index]['base']
        video, fps = video_loader(path)
        if video is None:
            print('skipping video {}'.format(path))
            return self[index+1]
        if self.split == 'val_video':
            print('preparing video across {} locations'.format(self.test_gap))
            return [self.get_item(index, shift=t, video=video)
                    for t in np.linspace(0, 1.0, self.test_gap)]
        else:
            return self.get_item(index, shift=None, video=video)

    def get_item(self, index, shift=None, video=None):
        n = video.shape[0]
        if n-self.train_gap-2 <= 0:
            shift = 0
        elif shift is None:
            shift = np.random.randint(n-self.train_gap-2)
        elif shift > 1.0:
            pass
        else:
            shift = int(shift * (n-self.train_gap-2))
        img, target, meta = self._process_stack(video, shift, self.data['datas'][index])
        # img is n x h x w x c
        # target is n x nc
        return img, target, meta

    def __len__(self):
        return len(self.data['datas'])

    @classmethod
    def get(cls, args, splits=('train', 'val', 'val_video')):
        train_file = args.train_file
        val_file = args.val_file
        if 'train' in splits:
            train_dataset = cls(
                args, args.data, 'train', train_file, args.cache,
                transform=transforms.Compose([
                    videotransforms.RandomCrop(args.input_size),
                    videotransforms.RandomHorizontalFlip()
                ]),
                input_size=args.input_size)
        else:
            train_dataset = None
        if 'val' in splits:
            val_dataset = cls(
                args, args.valdata, 'val', val_file, args.cache,
                transform=transforms.Compose([
                    videotransforms.CenterCrop(args.input_size)
                ]),
                input_size=args.input_size)
        else:
            val_dataset = None
        if 'val_video' in splits:
            valvideo_dataset = cls(
                args, args.valdata, 'val_video', val_file, args.cache,
                transform=transforms.Compose([
                    videotransforms.CenterCrop(args.input_size)
                ]),
                input_size=args.input_size)
        else:
            valvideo_dataset = None
        return train_dataset, val_dataset, valvideo_dataset
