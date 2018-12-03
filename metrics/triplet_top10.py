from metrics.triplet_topk import TripletTopk


class TripletTop10(TripletTopk):
    def __init__(self):
        super(TripletTop10, self).__init__()
        self.k = 10
