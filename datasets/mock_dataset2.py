""" Define random data for quick debugging """
import torchvision
import torchvision.transforms as transforms
from datasets.dataset import Dataset


class MockDataset2(Dataset):
    def __init__(self, args, transform=None):
        self.data = torchvision.datasets.FakeData(
            transform=transform, num_classes=args.nclass, image_size=(3, args.input_size, args.input_size))
        self.test_gap = 25

    def __getitem__(self, index):
        im, target = self.data[index]
        meta = {}
        return im, target, meta

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return self.data.__repr__()

    @classmethod
    def get(cls, args):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = cls(args,
                            transform=transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                            ]))

        val_dataset = cls(args,
                          transform=transforms.Compose([
                              transforms.Resize(256),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              normalize,
                          ]))

        return train_dataset, val_dataset, val_dataset
