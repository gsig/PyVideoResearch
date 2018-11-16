""" Video loader for the Charades dataset """
from datasets.activitynet3 import ActivityNet3
from datasets.dataset_tsn import DatasetTSN


class ActivityNet3TSN(DatasetTSN, ActivityNet3):
    def __init__(self, *args, **kwargs):
        ActivityNet3.__init__(self, *args, **kwargs)
        DatasetTSN.__init__(self, *args, **kwargs)
