from metrics.triplet_topk_metric import TripletTopkMetric


class TripletTop10Metric(TripletTopkMetric):
    def __init__(self):
        super(TripletTop10Metric, self).__init__()
        self.k = 10
