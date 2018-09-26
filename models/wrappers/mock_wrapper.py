from wrapper import Wrapper


class MockWrapper(wrapper):
    def __init__(self, *args, *kwargs):
        super(Wrapper, self).__init__(*args, **kwargs)

    def forward(self, x):
        return self.basenet(x)
