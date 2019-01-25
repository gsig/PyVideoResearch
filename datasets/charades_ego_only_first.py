""" Dataset loader for the Charades dataset """
from datasets.charades import Charades
import copy


class CharadesEgoOnlyFirst(Charades):
    def __init__(self, *args, **kwargs):
        super(CharadesEgoOnlyFirst, self).__init__(*args, **kwargs)

    @staticmethod
    def parse_charades_csv(filename):
        labels = Charades.parse_charades_csv(filename)
        labels = dict((k, v) for k, v in labels.items() if 'EGO' in k)
        return labels

    @classmethod
    def get(cls, args, scale=(0.8, 1.0), splits=('train', 'val', 'val_video')):
        if ';' in args.train_file:
            args = copy.deepcopy(args)
            vars(args).update({
                'train_file': args.train_file.split(';')[0],
                'val_file': args.val_file.split(';')[0],
                'data': args.data.split(';')[0]})
        return super(CharadesEgoOnlyFirst, cls).get(args, scale, splits)
