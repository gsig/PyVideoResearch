""" Overloading Torchvision transforms to operate on a list """

import torchvision.transforms as parents

class CenterCrop(parents.CenterCrop):
    def __init__(self, *args, **kwargs):
        super(CenterCrop, self).__init__(*args, **kwargs)
    def __call__(self, img):
        return [super(CenterCrop, self).__call__(im) for im in img]
        

class RandomCrop(parents.RandomCrop):
    def __init__(self, *args, **kwargs):
        super(RandomCrop, self).__init__(*args, **kwargs)
    def __call__(self, img):
        return [super(RandomCrop, self).__call__(im) for im in img]


class RandomResizedCrop(parents.RandomResizedCrop):
    def __init__(self, *args):
        super(RandomResizedCrop, self).__init__(*args)
    def __call__(self, img):
        return [super(RandomResizedCrop, self).__call__(im) for im in img]


class Resize(parents.Resize):
    def __init__(self, *args, **kwargs):
        super(Resize, self).__init__(*args, **kwargs)
    def __call__(self, img):
        return [super(Resize, self).__call__(im) for im in img]


class ToTensor(parents.ToTensor):
    def __init__(self, *args, **kwargs):
        super(ToTensor, self).__init__(*args, **kwargs)
    def __call__(self, img):
        return [super(ToTensor, self).__call__(im) for im in img]


class Normalize(parents.Normalize):
    def __init__(self, *args, **kwargs):
        super(Normalize, self).__init__(*args, **kwargs)
    def __call__(self, img):
        return [super(Normalize, self).__call__(im) for im in img]

