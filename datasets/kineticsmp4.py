""" Dataset loader for the Charades dataset """
import torch
import torchvision.transforms as transforms
from datasets.kinetics import Kinetics
from PIL import Image
import numpy as np
import videotransforms
import os
from datasets.utils import ffmpeg_video_loader as video_loader


class Kineticsmp4(Kinetics):
    def __init__(self, *args, **kwargs):
        self.trainGAP = 64
        self.testGAP = 10
        self.inputsize = kwargs['inputsize']
        super(Kineticsmp4, self).__init__(*args, **kwargs)

    def prepare(self, path, labels, split):
        datadir = path
        int2cls = dict([(y,x) for x,y in self.cls2int.items()])
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
            data['base'] = '{}/{}/{}_{:06d}_{:06d}'.format(
                datadir, name, vid, label['start'], label['end'])
            data['n'] = n
            data['labels'] = label
            data['id'] = vid
            if split == 'val_video':
                spacing = np.linspace(0, n-self.trainGAP-1, self.testGAP)
                for loc in spacing:
                    data['shift'] = loc
                    datas.append(data)
            else:
                datas.append(data)

        return {'datas': datas, 'split': split}

    def __getitem__(self, index):
        """
        args:
            index (int): index
        returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        try:
            ims = []
            tars = []
            meta = {}
            path = '{}.mp4'.format(self.data['datas'][index]['base'])
            try:
                video, fps = video_loader(path)
            except (TypeError, Exception) as e:
                print('failed to load video {}'.format(path))
                print(e)
                return self[np.random.randint(len(self))]
            n = int((self.data['datas'][index]['n']-1)*fps)
            if hasattr(self.data['datas'][index], 'shift'):
                print('using shift')
                shift = self.data['datas'][index]['shift']
            else:
                if n <= self.trainGAP+2:
                    shift = 0
                else:
                    shift = np.random.randint(n-self.trainGAP-2)

            resize = transforms.Resize(int(256./224*self.inputsize))
            spacing = np.arange(shift, shift+self.trainGAP)
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
        except Exception, e:
            print('uncought error in get')
            print(e)
            raise

    def __len__(self):
        return len(self.data['datas'])

    @classmethod
    def get(cls, args):
        train_file = args.train_file
        val_file = args.val_file
        train_dataset = cls(
            args.data, 'train', train_file, args.cache,
            transform=transforms.Compose([
                videotransforms.RandomCrop(args.inputsize),
                videotransforms.RandomHorizontalFlip()
            ]),
            inputsize=args.inputsize)
        val_dataset = cls(
            args.valdata, 'val', val_file, args.cache,
            transform=transforms.Compose([
                videotransforms.CenterCrop(args.inputsize)
            ]),
            inputsize=args.inputsize)
        valvideo_dataset = cls(
            args.valdata, 'val_video', val_file, args.cache,
            transform=transforms.Compose([
                videotransforms.CenterCrop(args.inputsize)
            ]),
            inputsize=args.inputsize)
        return train_dataset, val_dataset, valvideo_dataset
