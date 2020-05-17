
import json
from random import shuffle

class Dataset:
    def __init__(self, values = None, labels = None, weights = None, label_identity=None):
        self.values = values
        self.labels = labels
        self.weights = weights
        self.label_identity = label_identity

    def __iter__(self):
        return zip(self.values, self.labels)

    def __len__(self):
        return len(list(zip(self.values, self.labels)))

    def __getitem__(self, item):
        return list(zip(self.values, self.labels))[item]

    def split(self, ratio = 0.33):
        entries = list(zip(self.values, self.labels))
        shuffle(entries)

        idx = len(entries) - int(len(entries) * ratio)

        train = entries[:idx]
        test = entries[idx:]

        v_train = [t[0] for t in train]
        l_train = [t[1] for t in train]

        t_test = [t[0] for t in test]
        l_test = [t[1] for t in test]

        ds1 = self.__class__(v_train, l_train)
        ds2 = self.__class__(t_test, l_test)

        return ds1, ds2

    def store(self, p):
        with open(p, "w") as f:
            json.dump(dict(values = self.values,
                           labels = self.labels,
                           weights = self.weights,
                           label_identity = self.label_identity
                           ), f)

    def load(self, p):
        with open(p, "r") as f:
            data = json.load(f)
        self.values = data['values']
        self.labels = data['labels']
        self.weights = data['weights']
        self.label_identity = data['label_identity']
        return self


class ImageLabelDataset(Dataset):
    def __init__(self, paths = None, labels=None, weights = None, label_identity=None):
        super(ImageLabelDataset, self).__init__(paths, labels, weights, label_identity)


class ImageMaskDataset(Dataset):
    def __init__(self, images=None, masks=None,  weights = None, label_identity=None):
        super(ImageMaskDataset, self).__init__(images, masks, weights, label_identity)
