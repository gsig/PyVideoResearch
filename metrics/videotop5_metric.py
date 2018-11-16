from metrics.utils import AverageMeter, accuracy
from metrics.metric import Metric


class VideoTop5Metric(Metric):
    def __init__(self):
        self.am = AverageMeter()

    def update(self, prediction, target):
        if target.dim() == 3:
            target = target.max(dim=1)[0]
        prediction_video = prediction.max(dim=0, keepdim=True)[0]
        target_video = target.max(dim=0, keepdim=True)[0]
        prec1, = accuracy(prediction_video, target_video, topk=(5,))
        self.am.update(prec1.item())

    def __repr__(self):
        return 'VideoPrec@5 {am.val:.3f} ({am.avg:.3f})'.format(am=self.am)

    def compute(self):
        return ('videotop5', self.am.avg)
