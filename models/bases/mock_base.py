from base import Base
import nn


class MockBase(Base):
    @classmethod
    def get(cls, args):
        model = nn.Linear(args.input_size, args.nclass)
        return model
