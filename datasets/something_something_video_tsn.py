""" Dataset loader for the Charades dataset """
from datasets.something_something_webm import SomethingSomethingwebm
from datasets.charades_video_tsn import CharadesVideoTSN


class SomethingSomethingVideoTSN(SomethingSomethingwebm, CharadesVideoTSN):
    def __init__(self, opts, *args, **kwargs):
        self.segments = opts.temporal_segments
        if 'test_gap' not in kwargs:
            kwargs['test_gap'] = 25
        super(SomethingSomethingVideoTSN, self).__init__(opts, *args, **kwargs)

    def _get_one_image(self, index, shift):
        return super(SomethingSomethingVideoTSN, self).__getitem__(index, shift=shift)

    def __getitem__(self, index):
        return CharadesVideoTSN.__getitem__(self, index)
