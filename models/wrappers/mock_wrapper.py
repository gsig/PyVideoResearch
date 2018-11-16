from wrapper import Wrapper


class MockWrapper(Wrapper):
    def __init__(self, *args, **kwargs):
        super(Wrapper, self).__init__(*args, **kwargs)

    def forward(self, x, meta):
        return self.basenet(x)
