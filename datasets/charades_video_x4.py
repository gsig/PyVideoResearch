import torchvision.transforms as transforms
from datasets.charades_video import CharadesVideo
import datasets.video_transforms as videotransforms


class CharadesVideoX4(CharadesVideo):
    def __init__(self, *args, **kwargs):
        if 'train_gap' not in kwargs:
            kwargs['train_gap'] = 32
        if 'test_gap' not in kwargs:
            kwargs['test_gap'] = 50
        super(CharadesVideoX4, self).__init__(*args, **kwargs)

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
                args, args.data, 'val', val_file, args.cache,
                transform=transforms.Compose([
                    videotransforms.CenterCrop(args.input_size)
                ]),
                input_size=args.input_size)
        else:
            val_dataset = None
        if 'val_video' in splits:
            valvideo_dataset = cls(
                args, args.data, 'val_video', val_file, args.cache,
                transform=transforms.Compose([
                    videotransforms.CenterCrop(args.input_size)
                ]),
                input_size=args.input_size)
        else:
            valvideo_dataset = None
        return train_dataset, val_dataset, valvideo_dataset
