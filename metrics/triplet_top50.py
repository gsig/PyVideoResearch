from metrics.triplet_topk import TripletTopk


class TripletTop50(TripletTopk):
    def __init__(self):
        super(TripletTop50, self).__init__()
        self.k = 50
