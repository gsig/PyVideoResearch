"""
    Defines tasks for evaluation
"""

from tasks.alignment_task import AlignmentTask
from datasets.charades_ego_video_alignment import CharadesEgoVideoAlignment


class Alignment3dTask(AlignmentTask):
    def __init__(self, *args, **kwargs):
        super(Alignment3dTask, self).__init__(*args, **kwargs)

    @classmethod
    def run(cls, model, criterion, epoch, args):
        task = cls(model, epoch, args)
        loader = CharadesEgoVideoAlignment.get(args)
        return task.alignment(loader, model, epoch, args)
