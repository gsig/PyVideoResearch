""" Video loader for the Charades dataset """
from datasets.activitynet2 import ActivityNet2
from datasets.dataset_tsn2 import DatasetTSN2


class ActivityNet2TSN(DatasetTSN2, ActivityNet2):
    def __init__(self, *args, **kwargs):
        ActivityNet2.__init__(self, *args, **kwargs)
        DatasetTSN2.__init__(self, *args, **kwargs)
