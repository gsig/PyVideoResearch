""" Dataset loader for the Charades dataset """
from charades_video import CharadesVideo


class CharadesVideo5(CharadesVideo):
    def __init__(self, *args, **kwargs):
        kwargs['test_gap'] = 5
        super(CharadesVideo5, self).__init__(*args, **kwargs)
