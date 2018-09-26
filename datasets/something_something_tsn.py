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

    def __getitem__(self, index):
        path = self.data['datas'][index]['base']
        try:
            video, fps = video_loader(path)
        except (TypeError, Exception) as e:
            print('failed to load video {}'.format(path))
            print(e)
            #return self[np.random.randint(len(self))]
            return self[index+1]

        n = video.shape[0]
        if hasattr(self.data['datas'][index], 'shift'):
            print('using shift')
            shift = self.data['datas'][index]['shift']
            shift = int(shift * (n-1))
        else:
            if n <= self.train_gap+2:
                shift = 0
            else:
                shift = np.random.randint(n)
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
        return imgs, targets, metas

    @classmethod
    def get(cls, args):
        return super(CharadesVideo, cls).get(args)
