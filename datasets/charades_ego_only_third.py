""" Dataset loader for the Charades dataset """
from datasets.charades_ego_only_first import CharadesEgoOnlyFirst
from datasets.charades import Charades


class CharadesEgoOnlyThird(CharadesEgoOnlyFirst, Charades):
    def __init__(self, *args, **kwargs):
        super(CharadesEgoOnlyThird, self).__init__(*args, **kwargs)

    @staticmethod
    def parse_charades_csv(filename):
        labels = Charades.parse_charades_csv(filename)
        labels = [x for x in labels if 'EGO' not in x['id']]
        return labels
