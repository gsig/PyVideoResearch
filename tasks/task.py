import inspect

class Task(object):
    """
    Class that defines a task
    """
    def __init__(self, *args, **kwargs):
        self.name = inspect.getfile(self.__class__).split('/')[-1].split('.')[0]

    @classmethod
    def run(cls, model, criterion, epoch, args):
        raise NotImplementedError
