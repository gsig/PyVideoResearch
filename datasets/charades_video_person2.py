""" Dataset loader for the Charades dataset """
import torch
import torchvision.transforms as transforms
import numpy as np
from glob import glob
from datasets.utils import default_loader
from charades_video import CharadesVideo


def _cropimg(img, x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    newx1 = int(max(min(x1-.5*w, img.size[1]), 0))
    newx2 = int(max(min(x2+.5*w, img.size[1]), 0))
    newy1 = int(max(min(y1-.0*h, img.size[0]), 0))
    newy2 = int(max(min(y2+.0*h, img.size[0]), 0))
    out = img.crop((newx1, newy1, newx2, newy2))
    return out


class CharadesVideoPerson2(CharadesVideo):
    def __init__(self, *args, **kwargs):
        self.train_gap = 64
        if 'test_gap' not in kwargs:
            kwargs['test_gap'] = 10
        self.input_size = 224
        super(CharadesVideoPerson2, self).__init__(*args, **kwargs)

    def _prepare(self, path, labels, split):
        datadir = path
        datas = []

        for i, (vid, label) in enumerate(labels.items()):
            iddir = datadir + '/' + vid
            persondir = '/scratch/gsigurds/anonymous_persons/' + vid
            lines = glob(iddir+'/*.jpg')
            personlines = glob(persondir+'/*.txt')
            persontimes = [int(x.split('-')[-1].split('.')[0]) for x in personlines]
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
            data['personbase'] = '{}/{}-'.format(persondir, vid)
            data['n'] = n
            data['labels'] = label
            data['id'] = vid
            data['persontimes'] = persontimes
            if split == 'val_video':
                spacing = np.linspace(0, n-self.train_gap-1, self.test_gap)
                for loc in spacing:
                    data['shift'] = loc
                    datas.append(data)
            else:
                datas.append(data)
        return {'datas': datas, 'split': split}

    def __getitem__(self, index):
        ims = []
        personims = []
        tars = []
        meta = {}
        fps = 24
        n = self.data['datas'][index]['n']
        if hasattr(self.data['datas'][index], 'shift'):
            print('using shift')
            shift = self.data['datas'][index]['shift']
        else:
            shift = np.random.randint(n-self.train_gap-2)

        resize = transforms.Resize((int(256./224*self.input_size), int(256./224*self.input_size)))
        spacing = np.arange(shift, shift+self.train_gap)
        if len(self.data['datas'][index]['persontimes']) > 0:
            closest = np.argmin([abs(x-(shift+self.train_gap/2)) for x in self.data['datas'][index]['persontimes']])
            closest = self.data['datas'][index]['persontimes'][closest]
            with open(self.data['datas'][index]['personbase']+'{:06d}.txt'.format(closest), 'r') as f:
                box = [float(x) for x in f.readline().strip().split(' ')]
        else:
            print('no box, skipping cropping: {}'.format(self.data['datas'][index]['personbase']))
            box = []

        for loc in spacing:
            ii = int(np.floor(loc))
            path = '{}{:06d}.jpg'.format(self.data['datas'][index]['base'], ii+1)
            try:
                img = default_loader(path)
            except Exception as e:
                print('failed to load image {}'.format(path))
                print(e)
                raise
            if len(box) > 0:
                try:
                    personimg = _cropimg(img, *box[1:]).copy()
                    personimg = resize(personimg)
                    personimg = transforms.ToTensor()(personimg)
                    personimg = 2*personimg - 1
                    personims.append(personimg)
                except Exception, e:
                    print(e)
                    box = []
            img = resize(img)
            img = transforms.ToTensor()(img)
            img = 2*img - 1
            ims.append(img)
            if not len(box) > 0:
                personims.append(img)
            target = torch.IntTensor(self.num_classes).zero_()
            for x in self.data['datas'][index]['labels']:
                if x['start'] < ii/float(fps) < x['end']:
                    target[self.cls2int(x['class'])] = 1
            tars.append(target)
        meta['id'] = self.data['datas'][index]['id']
        meta['time'] = shift
        #img = torch.stack(personims).permute(0, 2, 3, 1).numpy()
        img = torch.stack(ims+personims).permute(0, 2, 3, 1).numpy()
        target = torch.stack(tars)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        #img = img.transpose([3, 0, 1, 2])
        # batch will be b x n x h x w x c
        # target will be b x n x nc
        return img, target, meta
