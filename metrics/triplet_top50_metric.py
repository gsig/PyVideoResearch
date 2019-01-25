from metrics.triplet_topk_metric import TripletTopkMetric


class TripletTop50Metric(TripletTopkMetric):
    def __init__(self):
        super(TripletTop50Metric, self).__init__()
        self.k = 50
