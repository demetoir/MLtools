from __future__ import division
from tensorflow.examples.tutorials.mnist import input_data
from script.data_handler.Base.BaseDataset import BaseDataset
from script.data_handler.Base.BaseDatasetPack import BaseDatasetPack
import numpy as np

Xs = 'Xs'
Xs_img = 'Xs_img'
Ys = 'Ys'
TRAIN_SIZE = 60000
TEST_SIZE = 10000
LABEL_SIZE = 10


class MNIST_train(BaseDataset):
    SIZE = TRAIN_SIZE
    LABEL_SIZE = LABEL_SIZE
    Xs_img = Xs_img
    BATCH_KEYS = [
        Xs,
        Ys,
        Xs_img
    ]

    def load(self, path, limit=None):
        mnist = input_data.read_data_sets(path, one_hot=True)
        self._data[Xs], self._data[Ys] = mnist.train.next_batch(self.SIZE)

    def save(self):
        pass

    def transform(self):
        data = self._data[Xs]
        shape = data.shape
        data = np.reshape(data, [shape[0], 28, 28, 1])
        self._data[Xs] = data


class MNIST_test(BaseDataset):
    SIZE = TEST_SIZE
    LABEL_SIZE = LABEL_SIZE
    Xs_img = Xs_img
    BATCH_KEYS = [
        Xs,
        Ys,
        Xs_img
    ]

    def load(self, path, limit=None):
        mnist = input_data.read_data_sets(path, one_hot=True)

        self._data[Xs], self._data[Ys] = mnist.test.next_batch(self.SIZE)

    def save(self):
        pass

    def transform(self):
        data = self._data[Xs]
        shape = data.shape
        data = np.reshape(data, [shape[0], 28, 28, 1])
        self._data[Xs] = data


class MNIST(BaseDatasetPack):
    def __init__(self):
        super().__init__()
        self.pack['train'] = MNIST_train()
        self.pack['test'] = MNIST_test()
