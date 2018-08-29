from __future__ import division
from tensorflow.examples.tutorials.mnist import input_data
from script.data_handler.Base.BaseDataset import BaseDataset
from script.data_handler.Base.BaseDatasetPack import BaseDatasetPack
import numpy as np

TRAIN_SIZE = 60000
TEST_SIZE = 10000
LABEL_SIZE = 10


class MNIST_train(BaseDataset):
    SIZE = TRAIN_SIZE
    LABEL_SIZE = LABEL_SIZE

    def load(self, path):
        mnist = input_data.read_data_sets(path, one_hot=True)
        x, y = mnist.train.next_batch(self.SIZE)
        self.add_data('x', x)
        self.add_data('y', y)

        data = self._data['x']
        shape = data.shape
        data = np.reshape(data, [shape[0], 28, 28, 1])
        self._data['x'] = data
        self.x_keys = ['x']
        self.y_keys = ['y']


class MNIST_test(BaseDataset):
    SIZE = TEST_SIZE
    LABEL_SIZE = LABEL_SIZE

    def load(self, path):
        mnist = input_data.read_data_sets(path, one_hot=True)

        x, y = mnist.test.next_batch(self.SIZE)
        self.add_data('x', x)
        self.add_data('y', y)

        data = self._data['x']
        shape = data.shape
        data = np.reshape(data, [shape[0], 28, 28, 1])
        self._data['x'] = data
        self.x_keys = ['x']
        self.y_keys = ['y']


class MNIST(BaseDatasetPack):
    def __init__(self):
        super().__init__()
        self.pack['train'] = MNIST_train()
        self.pack['test'] = MNIST_test()
