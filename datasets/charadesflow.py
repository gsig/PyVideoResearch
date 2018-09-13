""" Dataset loader for the Charades dataset """
import torch
import torchvision.transforms as transforms
import transforms as arraytransforms
from charades import Charades, cls2int
from PIL import Image
import numpy as np
from glob import glob


def default_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')


class Charadesflow(Charades):
    def __init__(self, *args, **kwargs):
        super(Charadesflow,self).__init__(*args, **kwargs)
        
    def prepare(self, path, labels, split):
        FPS, GAP, testGAP = 24, 4, 25
        STACK=10
        datadir = path
        image_paths, targets, ids = [], [], []

        for i, (vid, label) in enumerate(labels.iteritems()):
            iddir = datadir + '/' + vid
            lines = glob(iddir+'/*.jpg')
            n = len(lines)/2
            if i % 100 == 0:
                print("{} {}".format(i, iddir))
            if n == 0:
                continue
            if split == 'val_video':
                target = torch.IntTensor(157).zero_()
                for x in label:
                    target[cls2int(x['class'])] = 1
                spacing = np.linspace(0, n-1-STACK-1, testGAP)  # fit 10 optical flow pairs
                for loc in spacing:
                    impath = '{}/{}-{:06d}x.jpg'.format(
                        iddir, vid, int(np.floor(loc))+1)
                    image_paths.append(impath)
                    targets.append(target)
                    ids.append(vid)
            else:
                for x in label:
                    for ii in range(0, n-1, GAP):
                        if x['start'] < ii/float(FPS) < x['end']:
                            if ii>n-1-STACK-1: continue  # fit 10 optical flow pairs
                            impath = '{}/{}-{:06d}x.jpg'.format(
                                iddir, vid, ii+1)
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
        base = path[:-5-6]
        framenr = int(path[-5-6:-5])
        assert '{}{:06d}x.jpg'.format(base,framenr) == path
        STACK=10
        img = []
        for i in range(STACK):
            x = '{}{:06d}x.jpg'.format(base,framenr+i)
            y = '{}{:06d}y.jpg'.format(base,framenr+i)
            imgx = default_loader(x)
            imgy = default_loader(y)
            img.append(imgx)
            img.append(imgy)
        target = self.data['targets'][index]
        meta = {}
        meta['id'] = self.data['ids'][index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, meta

    @classmethod
    def get(cls, args):
        """ Entry point. Call this function to get all Charades dataloaders """
        normalize = arraytransforms.Normalize(mean=[0.502], std=[1.0])
        train_file = args.train_file
        val_file = args.val_file
        train_dataset = cls(
            args.data, 'train', train_file, args.cache,
            transform=transforms.Compose([
                arraytransforms.RandomResizedCrop(224),
                arraytransforms.ToTensor(),
                normalize,
                transforms.Lambda(lambda x: torch.cat(x)),
            ]))
        val_transforms = transforms.Compose([
                arraytransforms.Resize(256),
                arraytransforms.CenterCrop(224),
                arraytransforms.ToTensor(),
                normalize,
                transforms.Lambda(lambda x: torch.cat(x)),
            ])
        val_dataset = cls(
            args.data, 'val', val_file, args.cache, transform=val_transforms)
        valvideo_dataset = cls(
            args.data, 'val_video', val_file, args.cache, transform=val_transforms)
        return train_dataset, val_dataset, valvideo_dataset
