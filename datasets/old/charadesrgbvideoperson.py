""" Dataset loader for the Charades dataset """
import torch
import torchvision.transforms as transforms
import numpy as np
from glob import glob
from charadesrgb import Charades, cls2int, default_loader
import videotransforms


def cropimg(img, x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    newx1 = int(max(min(x1-.5*w, img.shape[3]), 0))
    newx2 = int(max(min(x2+.5*w, img.shape[3]), 0))
    newy1 = int(max(min(y1-.0*h, img.shape[2]), 0))
    newy2 = int(max(min(y2+.0*h, img.shape[2]), 0))
    return img[newx1:newy1, newx2:newy2, :]


class CharadesVideoPerson(Charades):
    def __init__(self, *args, **kwargs):
        self.trainGAP = 64
        self.testGAP = 25
        self.inputsize = 224
        super(CharadesVideoPerson, self).__init__(*args, **kwargs)

    def prepare(self, path, labels, split):
        datadir = path
        datas = []

        for i, (vid, label) in enumerate(labels.iteritems()):
            iddir = datadir + '/' + vid
            persondir = '/scratch/gsigurds/anonymous_person/' + vid
            lines = glob(iddir+'/*.jpg')
            personlines = glob(persondir+'/*.txt')
            persontimes = [int(x.split('-')[-1].split('.')[0]) for x in personlines]
            n = len(lines)
            if i % 100 == 0:
                print("{} {}".format(i, iddir))
            if n == 0:
                print('empty: {}'.format(iddir))
                continue
            if n <= self.trainGAP + 1:
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
                spacing = np.linspace(0, n-self.trainGAP-1, self.testGAP)
                for loc in spacing:
                    data['shift'] = loc
                    datas.append(data)
            else:
                datas.append(data)
        return {'datas': datas, 'split': split}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        try:
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
                shift = np.random.randint(n-self.trainGAP-2)

            resize = transforms.Resize(int(256./224*self.inputsize))
            spacing = np.arange(shift, shift+self.trainGAP)
            if len(self.data['datas'][index]['persontimes']) > 0:
                closest = np.argmin([abs(x-(shift+self.trainGAP/2)) for x in self.data['datas'][index]['persontimes']])
                closest = self.data['datas'][index]['persontimes'][closest]
                with open(self.data['datas'][index]['persontimes']+'{:06d}.jpg'.format(closest), 'r') as f:
                    box = [float(x) for x in f.readline().strip().split(' ')]
            else:
                box = []

            for loc in spacing:
                ii = int(np.floor(loc))
                path = '{}{:06d}.jpg'.format(self.data['datas'][index]['base'], ii+1)
                try:
                    img = default_loader(path)
                except Exception, e:
                    print('failed to load image {}'.format(path))
                    print(e)
                    raise
                if len(box) > 0:
                    personimg = cropimg(img, *box[1:]).copy()
                    personimg = resize(personimg)
                    personimg = transforms.ToTensor()(personimg)
                    personimg = 2*personimg - 1
                    personims.append(personimg)
                img = resize(img)
                img = transforms.ToTensor()(img)
                img = 2*img - 1
                ims.append(img)
                if not len(box) > 0:
                    personims.append(img)
                target = torch.IntTensor(self.num_classes).zero_()
                for x in self.data['datas'][index]['labels']:
                    if x['start'] < ii/float(fps) < x['end']:
                        target[cls2int(x['class'])] = 1
                tars.append(target)
            meta['id'] = self.data['datas'][index]['id']
            meta['time'] = shift
            #img = torch.stack(ims).permute(0, 2, 3, 1).numpy()
            img = torch.stack(ims+personims).permute(0, 2, 3, 1).numpy()
            #target = torch.stack(tars)
            target = torch.stack(tars+tars) #TODO correct?
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
        return len(self.data['datas'])


def get(args):
    train_file = args.train_file
    val_file = args.val_file
    train_dataset = CharadesVideoPerson(
        args.data, 'train', train_file, args.cache,
        transform=transforms.Compose([
            videotransforms.RandomCrop(224),
            videotransforms.RandomHorizontalFlip()
        ]))
    val_dataset = CharadesVideoPerson(
        args.data, 'val', val_file, args.cache,
        transform=transforms.Compose([
            videotransforms.CenterCrop(224)
        ]))
    valvideo_dataset = CharadesVideoPerson(
        args.data, 'val_video', val_file, args.cache,
        transform=transforms.Compose([
            videotransforms.CenterCrop(224)
        ]))
    return train_dataset, val_dataset, valvideo_dataset
