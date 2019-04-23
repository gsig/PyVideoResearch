""" Dataset loader for the Charades dataset """
import torchvision.transforms as transforms
from datasets.kinetics_mp4 import Kineticsmp4
import datasets.video_transforms as videotransforms


class Kineticsmp4X2(Kineticsmp4):
    def __init__(self, *args, **kwargs):
        super(Kineticsmp4X2, self).__init__(*args, **kwargs)

    def get_item(self, index, shift=None, video=None):
        img, target, meta = super(Kineticsmp4X2, self).get_item(index, shift=shift, video=video)
        img = img[::2, :, :, :]
        target = target[::2, :]
        return img, target, meta

    @classmethod
    def get(cls, args, splits=('train', 'val', 'val_video')):
        train_file = args.train_file
        val_file = args.val_file
        if 'train' in splits:
            train_dataset = cls(
                args, args.data, 'train', train_file, args.cache,
                transform=transforms.Compose([
                    videotransforms.RandomCrop(args.input_size),
                    videotransforms.RandomHorizontalFlip()
                ]),
                input_size=args.input_size)
        else:
            train_dataset = None
        if 'val' in splits:
            val_dataset = cls(
                args, args.valdata, 'val', val_file, args.cache,
                #transform=transforms.Compose([
                #    #videotransforms.RandomCrop(args.input_size),
                #    #videotransforms.RandomHorizontalFlip()
                #    videotransforms.CenterCrop(256)
                #]),
                input_size=args.input_size)
        else:
            val_dataset = None
        if 'val_video' in splits:
            valvideo_dataset = cls(
                args, args.valdata, 'val_video', val_file, args.cache,
                #transform=transforms.Compose([
                #    videotransforms.CenterCrop(256)
                #]),
                input_size=args.input_size)
        else:
            valvideo_dataset = None
        return train_dataset, val_dataset, valvideo_dataset
