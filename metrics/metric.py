class Metric(object):
    """
    Class that defines a performance metric
    """
    def __init__(self):
        pass

    def update(self, prediction, target):
        """
        Add prediction and target to this metric 
        """
        raise NotImplementedError

    def __repr__(self):
        """
        Optionally return running information about this metric 
        """
        return self.__class__.__name__

    def compute(self):
        """
        Returns a tuple of the name of the metric and the final computed value of the metric
        """
        raise NotImplementedError
