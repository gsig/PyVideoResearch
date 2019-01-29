from datasets.charades_video import CharadesVideo


class CharadesVideoSfm(CharadesVideo):
    def __init__(self, *args, **kwargs):
        if 'train_gap' not in kwargs:
            kwargs['train_gap'] = 2
        super(CharadesVideoSfm, self).__init__(*args, **kwargs)
