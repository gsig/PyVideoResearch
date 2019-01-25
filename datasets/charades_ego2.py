""" Dataset loader for the Charades dataset """
from datasets.charades_ego import CharadesEgo
import torchvision.transforms as transforms


class CharadesEgo2(CharadesEgo):
    def __init__(self, *args, **kwargs):
        super(CharadesEgo2, self).__init__(*args, **kwargs)

    @classmethod
    def get(cls, args, scale=(0.8, 1.0), splits=('train', 'val', 'val_video')):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if 'train' in splits:
            train_dataset = cls(
                args, args.data, 'train', args.train_file, args.cache,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(args.input_size, scale),
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4),
                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),  # missing PCA lighting jitter
                    normalize,
                ]))
        else:
            train_dataset = None
        if 'val' in splits:
            val_dataset = cls(
                args, args.data, 'val', args.val_file, args.cache,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(args.input_size, scale),
                    transforms.ToTensor(),
                    normalize,
                ]))
        else:
            val_dataset = None
        if 'val_video' in splits:
            valvideo_dataset = cls(
                args, args.data, 'val_video', args.val_file, args.cache,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(args.input_size, scale),
                    transforms.ToTensor(),
                    normalize,
                ]))
        else:
            valvideo_dataset = None
        return train_dataset, val_dataset, valvideo_dataset
