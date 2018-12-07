"""
    Dataset loader that combines Charadesego and Charades
    train and val loaders are concatenated
    Charades includes a [x, x, x] transform to match the input to charadesego
    labels for Charades are negative to distinguish between them
"""
import torchvision.transforms as transforms
from torch.utils.data.dataset import ConcatDataset
from datasets.charades_video import CharadesVideo
from datasets.charades_ego_video import CharadesEgoVideo
import copy
import torch


class CharadesVideoMeta(CharadesVideo):
    def __init__(self, *args, **kwargs):
        super(CharadesVideoMeta, self).__init__(*args, **kwargs)

    def get_item(self, index, shift=None):
        ims, target, meta = super(CharadesVideoMeta, self).get_item(index, shift)
        meta.update({'thirdtime': 0,
                     'firsttime_pos': 0,
                     'firsttime_neg': 0,
                     'n': 0,
                     'n_ego': 0,
                     })
        return [ims, ims, ims], target, meta


class CharadesEgoVideoMeta(CharadesEgoVideo):
    def __init__(self, *args, **kwargs):
        super(CharadesEgoVideoMeta, self).__init__(*args, **kwargs)

    def get_item(self, index, shift=None):
        ims, target, meta = super(CharadesEgoVideoMeta, self).get_item(index, shift)
        newtarget = torch.IntTensor(self.num_classes).zero_()
        newtarget[0] = target
        meta['time'] = 0
        return ims, newtarget, meta


class CharadesEgoVideoPlusCharades(CharadesVideoMeta):
    @classmethod
    def get(cls, args):
        newargs = copy.deepcopy(args)
        vars(newargs).update({
            'train_file': args.train_file.split(';')[1],
            'val_file': args.val_file.split(';')[1],
            'data': args.data.split(';')[1]})
        vars(args).update({
            'train_file': args.train_file.split(';')[0],
            'val_file': args.val_file.split(';')[0],
            'data': args.data.split(';')[0]})

        train_datasetego, val_datasetego, _ = CharadesEgoVideoMeta.get(args)
        train_dataset, val_dataset, valvideo_dataset = super(CharadesEgoVideoPlusCharades, cls).get(newargs)

        train_dataset.target_transform = transforms.Lambda(lambda x: -x)
        val_dataset.target_transform = transforms.Lambda(lambda x: -x)

        train_dataset = ConcatDataset([train_dataset] + [train_datasetego] * 3)  # magic number to balance
        val_dataset = ConcatDataset([val_dataset] + [val_datasetego] * 3)
        return train_dataset, val_dataset, valvideo_dataset
