""" Dataset loader for the Charades dataset """
from charades_video import CharadesVideo


class CharadesVideo10(CharadesVideo):
    def __init__(self, *args, **kwargs):
        kwargs['test_gap'] = 10
        super(CharadesVideo10, self).__init__(*args, **kwargs)
