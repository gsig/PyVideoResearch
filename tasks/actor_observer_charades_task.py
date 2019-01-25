from datasets.get import get_dataset
from tasks.video_task import VideoTask
from models.wrappers.actor_observer_classifier_wrapper import ActorObserverClassifierWrapper
from models.utils import set_distributed_backend
from models.criteria.default_criterion import DefaultCriterion
import copy


class ActorObserverCharadesTask(VideoTask):
    def __init__(self, model, epoch, args):
        super(ActorObserverCharadesTask, self).__init__(model, epoch, args)

    @classmethod
    def run(cls, model, criterion, epoch, args):
        model = ActorObserverClassifierWrapper(model, args)
        model = set_distributed_backend(model, args)
        criterion = DefaultCriterion(args)
        task = cls(model, epoch, args)
        newargs = copy.deepcopy(args)
        if ';' in args.train_file:
            vars(newargs).update({
                'train_file': args.train_file.split(';')[1],
                'val_file': args.val_file.split(';')[1],
                'data': args.data.split(';')[1]})
        if '3d' in args.arch:
            loader, = get_dataset(newargs, splits=('val_video', ), dataset='charades_video')
        else:
            loader, = get_dataset(newargs, splits=('val_video', ), dataset='charades')
        return task.validate_video(loader, model, criterion, epoch, args)
