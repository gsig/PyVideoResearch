from metrics.triplet_topk_metric import TripletTopkMetric


class TripletTop5Metric(TripletTopkMetric):
    def __init__(self):
        super(TripletTop5Metric, self).__init__()
        self.k = 5
