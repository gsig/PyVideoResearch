class Base(object):
    @classmethod
    def get(cls, args):
        """ Call this function to get the model 
            Returns a child of nn.Module
        """
        raise NotImplementedError()
