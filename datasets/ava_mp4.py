""" Dataset loader for the Charades dataset """
from __future__ import division
from datasets.kinetics_mp4 import Kineticsmp4
from datasets.dataset import Dataset
from datasets.utils import cache
from datasets.utils import ffmpeg_video_loader as video_loader
import torchvision.transforms as transforms
import datasets.video_transforms as videotransforms
import csv
import numpy as np
from collections import defaultdict
import torch
from PIL import Image


class AVAmp4(Kineticsmp4, Dataset):
    def __init__(self, args, root, split, labelpath, cachedir,
                 transform=None, target_transform=None, input_size=224, test_gap=10):
        Dataset.__init__(self, test_gap, split)
        self.num_classes = 80
        self.transform = transform
        self.target_transform = target_transform
        self.cls2int = dict((str(x+1), x) for x in range(80))
        self.labels = self.parse_ava_csv(labelpath, self.cls2int)
        self.root = root
        self.train_gap = 64
        self.input_size = input_size
        cachename = '{}/{}_{}.pkl'.format(cachedir,
                                          self.__class__.__name__, split)
        self._data = cache(cachename)(self._prepare)(root, self.labels, split)

    def _get_video_path(self, path, vid, label):
        return '{}/{}_{}_{}.mp4'.format(path, vid, 902, 905)

    def __getitem__(self, index):
        path = self.data['datas'][index]['base']
        base = path.replace('_902_905.mp4', '')
        label = self.data['datas'][index]['labels']
        mid_start_time = 902-3 + 3*int((label['start']-902+3)//3)
        path0 = '{}_{}_{}.mp4'.format(base, mid_start_time-3, mid_start_time)
        path1 = '{}_{}_{}.mp4'.format(base, mid_start_time, mid_start_time+3)
        path2 = '{}_{}_{}.mp4'.format(base, mid_start_time+3, mid_start_time+2*3)
        diff = label['start'] - mid_start_time
        video0, fps0 = video_loader(path0)
        video1, fps1 = video_loader(path1)
        video2, fps2 = video_loader(path2)
        if video0 is None or video1 is None or video2 is None:
            print('video is none, skipping')
            return self[index+1]
        video = np.concatenate([video0, video1, video2])
        shift = int(len(video0) + diff * fps1 - self.train_gap//2)
        if shift < 0:
            print('negative shift, skipping')
            return self[index+1]
        if self.split == 'val_video':
            assert False, 'val_video is not supported for AVA'
        else:
            img, target, meta = self.get_item(index, shift=shift, video=video)
            meta['start'] = label['start']
            meta['boxes'] = torch.Tensor(label['boxes'])
            meta['labels'] = torch.Tensor(label['labels']).long()
            #meta['pids'] = label['pids']
            if self.split == 'train' and np.random.rand() > .5:
                # data augmentation, hflip
                img = np.ascontiguousarray(img[::-1, :, :])
                meta['boxes'][:, [1, 3]] = 1 - meta['boxes'][:, [3, 1]]
            return img, target, meta

    def _process_stack(self, video, shift, data):
        # TODO remove this and standardize normalization
        ims, tars, meta = [], [], {}
        meta['do_not_collate'] = True
        if self.split == 'train' and np.random.random() > 0.5:
            resize = transforms.Resize(int(320./224*self.input_size))
        else:
            resize = transforms.Resize(int(256./224*self.input_size))
        #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                 std=[0.229, 0.224, 0.225])
        spacing = np.arange(shift, shift+self.train_gap)
        for loc in spacing:
            img = video[loc] if loc < len(video) else video[-1]
            img = resize(Image.fromarray(img))
            img = transforms.ToTensor()(img)
            img = 2*img - 1
            #img = normalize(img)
            ims.append(img)
            target = torch.IntTensor(self.num_classes).zero_()
            target[data['labels']['class']] = 1
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

    @staticmethod
    def parse_ava_csv(filename, cls2int):
        labels = {}
        allboxes = defaultdict(list)
        alllabels = defaultdict(list)
        allpids = defaultdict(list)
        with open(filename) as f:
            header = 'video_id,middle_frame_timestamp,x1,y1,x2,y2,action_id,person_id'
            reader = csv.DictReader(f, fieldnames=header.split(','))

            for row in reader:
                vid = row['video_id']
                label = row['action_id']
                pid = int(row['person_id'])
                start = int(row['middle_frame_timestamp'])
                labelnumber = cls2int[label]
                box = (float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2']))
                labels[(vid, start)] = {
                    'class': labelnumber,
                    'start': start,
                    'box': box,
                    'vid': vid,
                    'pid': pid,
                }
                allboxes[(vid, start)].append(box)
                alllabels[(vid, start)].append(labelnumber)
                allpids[(vid, start)].append(pid)
        for k, v in labels.items():
            v['boxes'] = allboxes[(v['vid'], v['start'])]
            v['labels'] = alllabels[(v['vid'], v['start'])]
            v['pids'] = allpids[(v['vid'], v['start'])]
        return labels

    @classmethod
    def get(cls, args, splits=('train', 'val', 'val_video')):
        train_file = args.train_file
        val_file = args.val_file
        if 'train' in splits:
            train_dataset = cls(
                args, args.data, 'train', train_file, args.cache,
                transform=transforms.Compose([
                    videotransforms.ScaledCenterCrop(args.input_size),
                ]),
                input_size=args.input_size)
        else:
            train_dataset = None
        if 'val' in splits:
            val_dataset = cls(
                args, args.valdata, 'val', val_file, args.cache,
                transform=transforms.Compose([
                    videotransforms.ScaledCenterCrop(args.input_size),
                ]),
                input_size=args.input_size)
        else:
            val_dataset = None
        if 'val_video' in splits:
            valvideo_dataset = cls(
                args, args.valdata, 'val_video', val_file, args.cache,
                transform=transforms.Compose([
                    videotransforms.ScaledCenterCrop(args.input_size),
                ]),
                input_size=args.input_size)
        else:
            valvideo_dataset = None
        return train_dataset, val_dataset, valvideo_dataset
