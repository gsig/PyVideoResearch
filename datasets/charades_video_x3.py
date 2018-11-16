import torchvision.transforms as transforms
from datasets.charades_video import CharadesVideo
import datasets.video_transforms as videotransforms


class CharadesVideoX3(CharadesVideo):
    def __init__(self, *args, **kwargs):
        if 'train_gap' not in kwargs:
            kwargs['train_gap'] = 128
        if 'test_gap' not in kwargs:
            kwargs['test_gap'] = 50
        super(CharadesVideoX3, self).__init__(*args, **kwargs)

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
            args, args.data, 'val', val_file, args.cache,
            transform=transforms.Compose([
                videotransforms.CenterCrop(288)
            ]),
            input_size=args.input_size)
        valvideo_dataset = cls(
            args, args.data, 'val_video', val_file, args.cache,
            transform=transforms.Compose([
                videotransforms.CenterCrop(288)
            ]),
            input_size=args.input_size)
        return train_dataset, val_dataset, valvideo_dataset
