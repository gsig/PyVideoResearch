""" Video loader for the Charades dataset """
from datasets.charades_video import CharadesVideo
from datasets.charades_video_tsn import CharadesVideoTSN
from datasets.utils import default_loader
import torch


class CharadesTSN(CharadesVideoTSN, CharadesVideo):
    def __init__(self, *args, **kwargs):
        super(CharadesTSN, self).__init__(*args, **kwargs)

    def _get_one_image(self, index, shift):
        fps = 24
        path = '{}{:06d}.jpg'.format(self.data['datas'][index]['base'], shift+1)
        target = torch.IntTensor(self.num_classes).zero_()
        for x in self.data['datas'][index]['labels']:
            if x['start'] < shift/float(fps) < x['end']:
                target[self.cls2int(x['class'])] = 1
        meta = {}
        meta['id'] = self.data['datas'][index]['id']
        meta['time'] = shift
        img = default_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, meta

    @classmethod
    def get(cls, args):
        return super(CharadesVideo, cls).get(args)
