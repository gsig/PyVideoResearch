""" Dataset loader for the Charades dataset """
import torchvision.transforms as transforms
from datasets.charades_ego import CharadesEgo
import torch

class CharadesEgoAlignment(CharadesEgo):
    @classmethod
    def get(cls, args):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        val_dataset = cls(
            args, args.data, 'val', args.val_file, args.cache,
            transform=transforms.Compose([
                transforms.Resize(int(256. / 224 * args.inputsize)),
                transforms.CenterCrop(args.inputsize),
                transforms.ToTensor(),
                normalize,
            ]))

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=1, pin_memory=True)

        return val_loader
