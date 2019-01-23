""" Dataset loader for the Charades dataset """
from datasets.charades_video import CharadesVideo
import copy


class CharadesEgoVideoOnlyFirst(CharadesVideo):
    def __init__(self, *args, **kwargs):
        super(CharadesEgoVideoOnlyFirst, self).__init__(*args, **kwargs)

    @staticmethod
    def parse_charades_csv(filename):
        labels = CharadesVideo.parse_charades_csv(filename)
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
        return super(CharadesEgoVideoOnlyFirst, cls).get(args, splits)
