""" Dataset loader for the Charades dataset """
import torchvision.transforms as transforms
from datasets.charades_ego_video import CharadesEgoVideo
import torch
import datasets.video_transforms as videotransforms
from glob import glob
import numpy as np
import copy


class CharadesEgoVideoAlignment(CharadesEgoVideo):
    def _prepare(self, path, labels, split):
        datadir = path
        datas = []

        for i, (vid, label) in enumerate(labels.items()):
            gap = 4
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
            spacing = range(0, n-self.train_gap-1, gap)
            for loc in spacing:
                ii = int(np.floor(loc))
                data = {}
                data['base'] = '{}/{}-'.format(iddir, vid)
                data['base_ego'] = '{}EGO/{}EGO-'.format(iddir, vid)
                data['n'] = n
                data['n_ego'] = n_ego
                data['labels'] = label
                data['id'] = vid
                data['shift'] = ii
                datas.append(data)
        return {'datas': datas, 'split': split}

    @classmethod
    def get(cls, args):
        if ';' in args.train_file:
            args = copy.deepcopy(args)
            vars(args).update({
                'train_file': args.train_file.split(';')[0],
                'val_file': args.val_file.split(';')[0],
                'data': args.data.split(';')[0]})
        val_dataset = cls(
            args, args.data, 'val', args.val_file, args.cache,
            transform=transforms.Compose([
                videotransforms.CenterCrop(args.input_size)
            ]),
            input_size=args.input_size)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.video_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        return val_loader
