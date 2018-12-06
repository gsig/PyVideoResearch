""" Dataset loader for the Charades dataset """
from datasets.charades_video import CharadesVideo
from datasets.utils import default_loader
from glob import glob
from random import choice
import torch
import numpy as np
import torchvision.transforms as transforms


def to_ego_time(thirdtime, egoscale):
    return int(round(thirdtime * egoscale))


def get_neg_time(egoii, n_ego, d):
    allframes = range(n_ego)
    candidates = [x for x in allframes if not egoii - d <= x <= egoii + d]
    if len(candidates) == 0:
        return None
    else:
        return choice(candidates)


class CharadesEgoVideo(CharadesVideo):
    def __init__(self, *args, **kwargs):
        self.fps = 24
        self.deltaneg = 10 * self.fps
        super(CharadesEgoVideo, self).__init__(*args, **kwargs)

    def _prepare(self, path, labels, split):
        datadir = path
        datas = []

        for i, (vid, label) in enumerate(labels.items()):
            if split == 'val_video':
                continue
            iddir = datadir + '/' + vid
            n = len(glob(iddir + '/*.jpg'))
            n_ego = len(glob(iddir + 'EGO/*.jpg'))
            if i % 100 == 0:
                print("{} {}".format(i, iddir))
            if n == 0 or n_ego == 0:
                continue
            if n <= self.train_gap + 1:
                print('small: {}'.format(iddir))
                continue
            if n_ego <= self.train_gap + 1:
                print('small ego: {}'.format(iddir))
                continue
            data = {}
            data['base'] = '{}/{}-'.format(iddir, vid)
            data['base_ego'] = '{}EGO/{}EGO-'.format(iddir, vid)
            data['n'] = n
            data['n_ego'] = n_ego
            data['labels'] = label
            data['id'] = vid
            datas.append(data)
        return {'datas': datas, 'split': split}

    def _process_stack(self, base, shift, data):
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
            ii = int(np.floor(loc))
            path = '{}{:06d}.jpg'.format(base, ii+1)
            try:
                img = default_loader(path)
            except Exception as e:
                print('failed to load image {}'.format(path))
                print(e)
                raise
            img = resize(img)
            img = transforms.ToTensor()(img)
            img = normalize(img)
            ims.append(img)
            target = torch.IntTensor(self.num_classes).zero_()
            for x in data['labels']:
                if x['start'] < ii/float(self.fps) < x['end']:
                    target[self.cls2int(x['class'])] = 1
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

    def get_item(self, index, shift=None):
        data = self.data['datas'][index]
        n = data['n']
        n_ego = data['n_ego']
        if shift is None:
            if 'shift' in data:
                shift = data['shift']
            else:
                shift = np.random.randint(n-self.train_gap-2)
        else:
            shift = int(shift * (n-self.train_gap-2))
        scale = (n_ego-self.train_gap-2) / float(n-self.train_gap-2)
        shift_ego = to_ego_time(shift, scale)
        shift_ego_neg = get_neg_time(shift_ego, n_ego-self.train_gap-2, self.deltaneg)
        if shift_ego_neg is None:
            raise Exception('deltaneg too big for video')
        img, _, _ = self._process_stack(data['base'], shift, data)
        img_pos, _, _ = self._process_stack(data['base_ego'], shift_ego, data)
        img_neg, _, _ = self._process_stack(data['base_ego'], shift_ego_neg, data)
        meta = {'id': data['id'],
                'thirdtime': shift,
                'firsttime_pos': shift_ego,
                'firsttime_neg': shift_ego_neg,
                'n': n,
                'n_ego': n_ego}
        target = torch.ones(1)
        return [img_pos, img, img_neg], target, meta

    def __len__(self):
        return len(self.data['datas'])
