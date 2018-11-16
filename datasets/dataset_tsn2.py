""" Video loader for the a TSN style dataset """
from datasets.dataset import Dataset
import numpy as np


class DatasetTSN2(Dataset):
    def __init__(self, opts, *args, **kwargs):
        self.segments = opts.temporal_segments

    def _get_one_image(self, index, shift):
        return super(DatasetTSN2, self).get_item(index, shift=shift)

    def get_item(self, index, shift=None):
        if shift is None:
            shift = np.random.rand()
        n = self.data['datas'][index]['n']
        #skip = self.train_gap/float(n)
        skip = self.fps/float(n)
        ss = [shift] + [min(1, max(0, np.random.uniform(shift-skip, shift+skip)))
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
