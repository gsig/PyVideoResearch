""" Dataset loader for the Charades dataset """
from datasets.charades import Charades
from datasets.utils import default_loader
from glob import glob
from random import choice, random
import numpy as np
import torch


def to_ego_time(thirdtime, n, n_ego):
    egoscale = (n_ego - 1) / float(n - 1)
    return int(round(thirdtime * egoscale))


def get_neg_time(egoii, n_ego, d):
    allframes = range(n_ego)
    candidates = [x for x in allframes if not egoii - d <= x <= egoii + d]
    if len(candidates) == 0:
        return None
    else:
        return choice(candidates)


class CharadesEgo(Charades):
    def __init__(self, *args, **kwargs):
        self.fps = 24
        self.deltaneg = 10 * self.fps
        super(CharadesEgo, self).__init__(*args, **kwargs)

    def _prepare(self, path, labels, split):
        datadir = path
        image_paths, targets, ids, meta, outlabels = [], [], [], [], []

        for i, (vid, label) in enumerate(labels.items()):
            gap = 4
            iddir = datadir + '/' + vid
            n = len(glob(iddir + '/*.jpg'))
            n_ego = len(glob(iddir + 'EGO/*.jpg'))
            if i % 100 == 0:
                print("{} {}".format(i, iddir))
            if n == 0 or n_ego == 0:
                continue
            if split == 'val_video':
                spacing = [0]
            else:
                spacing = range(0, n-1, gap)
            for loc in spacing:
                ii = int(np.floor(loc))
                imbase = '{}/{}-'.format(iddir, vid)
                egobase = '{}EGO/{}EGO-'.format(iddir, vid)
                image_paths.append((imbase, egobase))
                targets.append(1)
                ids.append(vid)
                outlabels.append(label)
                meta.append({'thirdtime': ii,
                             'n': n,
                             'n_ego': n_ego})
        return {'image_paths': image_paths,
                'targets': targets,
                'ids': ids,
                'meta': meta,
                'labels': outlabels,
                'split': split}

    def get_item(self, index, shift=None):
        meta = self.data['meta'][index]
        n = meta['n']
        n_ego = meta['n_ego']
        if shift is None:
            shift = meta['thirdtime']
        else:
            shift = int(shift * (n-1))
        imbase, egobase = self.data['image_paths'][index]
        egoii = to_ego_time(shift, n, n_ego)
        negii = get_neg_time(egoii, n_ego, self.deltaneg)
        if negii is None:
            raise Exception('deltaneg too big for video')
        meta['firsttime_pos'] = egoii
        meta['thirdtime'] = shift
        meta['firsttime_neg'] = negii
        egopath_pos = '{}{:06d}.jpg'.format(egobase, egoii+1)
        impath = '{}{:06d}.jpg'.format(imbase, shift+1)
        egopath_neg = '{}{:06d}.jpg'.format(egobase, negii+1)
        impaths = (egopath_pos, impath, egopath_neg)

        target = self.data['targets'][index]
        classtarget = torch.IntTensor(self.num_classes).zero_()
        for x in self.data['labels'][index]:
            if x['start'] < shift/float(self.fps) < x['end']:
                classtarget[self.cls2int(x['class'])] = 1
        meta['class_target'] = classtarget

        meta['id'] = self.data['ids'][index]
        ims = [default_loader(im) for im in impaths]
        if self.transform is not None:
            ims = [self.transform(im) for im in ims]
        if self.target_transform is not None:
            target = self.target_transform(target)
        #if random() > 0.5:  # TODO DEBUG
        #    ims[2], ims[0] = ims[0], ims[2]
        #    target = -1
        return ims, target, meta

    @classmethod
    def get(cls, args, splits):
        return super(CharadesEgo, cls).get(args, splits=splits, scale=(0.8, 1.0))
