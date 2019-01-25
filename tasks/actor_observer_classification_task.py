from datasets.get import get_dataset
from tasks.video_task import VideoTask
from models.wrappers.actor_observer_classifier_wrapper import ActorObserverClassifierWrapper
from models.utils import set_distributed_backend
from models.criteria.default_criterion import DefaultCriterion


class ActorObserverClassificationTask(VideoTask):
    def __init__(self, model, epoch, args):
        super(ActorObserverClassificationTask, self).__init__(model, epoch, args)

    @classmethod
    def run(cls, model, criterion, epoch, args):
        model = ActorObserverClassifierWrapper(model, args)
        model = set_distributed_backend(model, args)
        criterion = DefaultCriterion(args)
        task = cls(model, epoch, args)
        loader, = get_dataset(args, splits=('val_video', ), dataset=args.actor_observer_classification_task_dataset)
        return task.validate_video(loader, model, criterion, epoch, args)
