""" Dataset loader for the Charades dataset """
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from datasets.charadesrgb import default_loader, cls2int
import numpy as np
from glob import glob
import videotransforms


class KineticsVideo(data.Dataset):
    def __init__(self, *args, **kwargs):
        self.trainGAP = 64
        self.testGAP = 25
        self.inputsize = kwargs['inputsize']
        super(KineticsVideo, self).__init__(*args, **kwargs)

    def prepare(self, path, labels, split):
        datadir = path
        datas = []

        for i, (vid, label) in enumerate(labels.iteritems()):
            iddir = '{}/{}_{:06d}_{:06d}'.format(datadir, vid, label['start'], label['end'])
            lines = glob(iddir + '/*.jpg')
            n = len(lines)
            if i % 1000 == 0:
                print("{} {}".format(i, iddir))
            if n == 0:
                print('empty: {}'.format(iddir))
                continue
            if n <= self.trainGAP + 1:
                print('small: {}'.format(iddir))
                continue
            data = {}
            data['base'] = '{}/{}_'.format(iddir, vid)
            data['base'] = '{}/{}_{:06d}_{:06d}_'.format(
                iddir, vid, label['start'], label['end'])
            data['n'] = n
            data['labels'] = label
            data['id'] = vid
            if split == 'val_video':
                spacing = np.linspace(0, n-self.traingap-1, self.testgap)
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
            fps = 24
            n = self.data['datas'][index]['n']
            if hasattr(self.data['datas'][index], 'shift'):
                print('using shift')
                shift = self.data['datas'][index]['shift']
            else:
                shift = np.random.randint(n-self.traingap-2)

            resize = transforms.resize(int(256./224*self.inputsize))
            spacing = np.arange(shift, shift+self.traingap)
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
                img = transforms.totensor()(img)
                img = 2*img - 1
                ims.append(img)
                target = torch.inttensor(self.num_classes).zero_()
                for x in self.data['datas'][index]['labels']:
                    if x['start'] < ii/float(fps) < x['end']:
                        target[cls2int(x['class'])] = 1
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
        except Exception, e:
            print('uncought error in get')
            print(e)
            raise

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


def get(args):
    train_file = args.train_file
    val_file = args.val_file
    train_dataset = KineticsVideo(
        args.data, 'train', train_file, args.cache,
        transform=transforms.Compose([
            videotransforms.RandomCrop(args.inputsize),
            videotransforms.RandomHorizontalFlip()
        ]),
        inputsize=args.inputsize)
    val_dataset = KineticsVideo(
        args.data, 'val', val_file, args.cache,
        transform=transforms.Compose([
            videotransforms.CenterCrop(args.inputsize)
        ]),
        inputsize=args.inputsize)
    valvideo_dataset = KineticsVideo(
        args.data, 'val_video', val_file, args.cache,
        transform=transforms.Compose([
            videotransforms.CenterCrop(args.inputsize)
        ]),
        inputsize=args.inputsize)
    return train_dataset, val_dataset, valvideo_dataset
