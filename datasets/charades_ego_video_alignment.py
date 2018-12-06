""" Dataset loader for the Charades dataset """
import torchvision.transforms as transforms
from datasets.charades_ego_video import CharadesEgoVideo
import torch
import datasets.video_transforms as videotransforms


class CharadesEgoAlignment(CharadesEgoVideo):
    @classmethod
    def get(cls, args):
        val_dataset = cls(
            args, args.data, 'val', args.val_file, args.cache,
            transform=transforms.Compose([
                videotransforms.CenterCrop(args.input_size)
            ]),
            input_size=args.input_size)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=1, pin_memory=True)

        return val_loader
