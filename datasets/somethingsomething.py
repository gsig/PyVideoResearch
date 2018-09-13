""" Dataset loader for the Charades dataset """
from glob import glob
import json
from datasets.charades import Charades
from dataset.utils import cache


def _parse_something_labels(filename):
    with open(filename) as f:
        labels = json.load(f)
    return labels


def _parse_something_json(filename, cls2int):
    labels = {}
    with open(filename) as f:
        data = json.load(f)
    for row in data:
        vid = row['id']
        label = row['template'].replace('[', '').replace(']', '')
        labelnumber = cls2int[label]
        labels[vid] = {'class': labelnumber}
    return labels


class SomethingSomething(Charades):
    def __init__(self, root, split, labelpath, cachedir, transform=None, target_transform=None):
        self.num_classes = 174
        self.transform = transform
        self.target_transform = target_transform
        self.cls2int = _parse_something_labels('/nfs.yoda/gsigurds/somethingsomething/something-something-v2-labels.json')
        self.labels = _parse_something_json(labelpath, self.cls2int)
        self.root = root
        self.testGAP = 50
        cachename = '{}/{}_{}.pkl'.format(cachedir,
                                          self.__class__.__name__, split)
        self.data = cache(cachename)(self.prepare)(root, self.labels, split)

    def prepare(self, path, labels, split):
        FPS, GAP, testGAP = 24, 4, self.testGAP
        datadir = path
        image_paths, targets, ids, times = [], [], [], []

        for i, (vid, label) in enumerate(labels.iteritems()):
            iddir = datadir + '/' + vid
            lines = glob(iddir + '/*.jpg')
            n = len(lines)
            if i % 1000 == 0:
                print("{} {}".format(i, iddir))
            if n == 0:
                continue
            if split == 'val_video':
                target = torch.IntTensor(self.num_classes).zero_()
                #for x in label:
                target[int(label['class'])] = 1
                spacing = np.linspace(0, n - 1, testGAP)
                for loc in spacing:
                    impath = '{}/{}-{:06d}.jpg'.format(
                        iddir, vid, int(np.floor(loc)) + 1)
                    image_paths.append(impath)
                    targets.append(target)
                    ids.append(vid)
                    times.append(int(np.floor(loc)) + 1)
            else:
                for ii in range(0, n - 1, GAP):
                    target = torch.IntTensor(self.num_classes).zero_()
                    #for x in label:
                    #    if x['start'] < ii/float(FPS) < x['end']:
                    target[int(label['class'])] = 1
                    impath = '{}/{}-{:06d}.jpg'.format(
                        iddir, vid, ii + 1)
                    image_paths.append(impath)
                    targets.append(target)
                    ids.append(vid)
                    times.append(ii)
        return {'image_paths': image_paths, 'targets': targets, 'ids': ids, 'times': times}


