from metrics.triplet_topk import TripletTopk


class TripletTop5(TripletTopk):
    def __init__(self):
        super(TripletTop5, self).__init__()
        self.k = 5
