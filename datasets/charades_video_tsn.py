""" Video loader for the Charades dataset """
import torch
from datasets.charades_video import CharadesVideo
import numpy as np


class CharadesVideoTSN(CharadesVideo):
    def __init__(self, opts, *args, **kwargs):
        self.segments = opts.temporal_segments
        if 'test_gap' not in kwargs:
            kwargs['test_gap'] = 25
        super(CharadesVideoTSN, self).__init__(opts, *args, **kwargs)

    def _get_one_image(self, index, shift):
        return super(CharadesVideoTSN, self).__getitem__(index, shift=shift)

    def __getitem__(self, index):
        n = self.data['datas'][index]['n']
        if hasattr(self.data['datas'][index], 'shift'):
            print('using shift')
            shift = self.data['datas'][index]['shift']
        else:
            shift = np.random.randint(n-self.train_gap-2)
        ss = [shift] + [np.random.randint(n-self.train_gap-2)
                        for _ in range(self.segments-1)]
        imgs, targets, metas = [], [], []
        for s in sorted(ss):
            img, target, meta = self._get_one_image(index, shift=s)
            imgs.append(img)
            targets.append(target)
            metas.append(meta)
        imgs = np.stack(imgs)
        targets = np.stack(targets)
        return imgs, targets, metas
