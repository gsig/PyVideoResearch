""" Video loader for the Charades dataset """
from datasets.activitynet import ActivityNet
from datasets.dataset_tsn import DatasetTSN


class ActivityNetTSN(DatasetTSN, ActivityNet):
    def __init__(self, *args, **kwargs):
        ActivityNet.__init__(self, *args, **kwargs)
        DatasetTSN.__init__(self, *args, **kwargs)
