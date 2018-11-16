""" Video loader for the Charades dataset """
from datasets.something_something_webm import SomethingSomethingwebm
from datasets.charades_video import CharadesVideo
from datasets.utils import ffmpeg_video_loader as video_loader
import torch
import numpy as np
from PIL import Image


class SomethingSomethingTSN(SomethingSomethingwebm, CharadesVideo):
    def __init__(self, opts, *args, **kwargs):
        self.segments = opts.temporal_segments
        super(SomethingSomethingTSN, self).__init__(opts, *args, **kwargs)

    def get_item(self, index, shift=None, video=None):
        path = self.data['datas'][index]['base']
        n = video.shape[0]
        if shift is None:
            shift = np.random.randint(n)
        else:
            shift = int(shift * (n-1))
        imgs, targets, metas = [], [], []
        ss = [shift] + [np.random.randint(n)
                        for _ in range(self.segments-1)]

        for s in sorted(ss):
            meta = {}
            meta['id'] = self.data['datas'][index]['id']
            meta['time'] = shift
            img = video[s]
            img = Image.fromarray(img)
            target = torch.IntTensor(self.num_classes).zero_()
            target[self.data['datas'][index]['labels']['class']] = 1
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            imgs.append(img)
            targets.append(target)
            metas.append(meta)
        imgs = np.stack(imgs)
        targets = np.stack(targets)
        # batch will be b x n x h x w x c
        # target will be b x n x nc
        return imgs, targets, metas

    @classmethod
    def get(cls, args):
        return super(CharadesVideo, cls).get(args)
