""" Dataset loader for the Charades dataset """
import torch
import torchvision.transforms as transforms
from datasets.kinetics_mp4 import Kineticsmp4
from PIL import Image
import numpy as np
import datasets.video_transforms as videotransforms
import os
from datasets.utils import ffmpeg_video_loader as video_loader


class Kineticsmp4X(Kineticsmp4):
    def __init__(self, *args, **kwargs):
        super(Kineticsmp4X, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target, meta = super(Kineticsmp4X, self).__getitem__(*args, **kwargs)
        img = img[:, ::2, :, :, :]
        target = target[:, ::2, :]
        return img, target, meta

    @classmethod
    def get(cls, args):
        train_file = args.train_file
        val_file = args.val_file
        train_dataset = cls(
            args, args.data, 'train', train_file, args.cache,
            transform=transforms.Compose([
                videotransforms.RandomCrop(args.input_size),
                videotransforms.RandomHorizontalFlip()
            ]),
            input_size=args.input_size)
        val_dataset = cls(
            args, args.valdata, 'val', val_file, args.cache,
            transform=transforms.Compose([
                videotransforms.CenterCrop(256)
            ]),
            input_size=args.input_size)
        valvideo_dataset = cls(
            args, args.valdata, 'val_video', val_file, args.cache,
            transform=transforms.Compose([
                videotransforms.CenterCrop(256)
            ]),
            input_size=args.input_size)
        return train_dataset, val_dataset, valvideo_dataset
