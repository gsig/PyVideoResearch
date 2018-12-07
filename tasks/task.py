class Task(object):
    """
    Class that defines a task
    """
    def __init__(self):
        pass

    @classmethod
    def run(cls, model, criterion, epoch, args):
        raise NotImplementedError
