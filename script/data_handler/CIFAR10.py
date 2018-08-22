from script.util.numpy_utils import np_imgs_NCWH_to_NHWC, np_index_to_onehot
from script.data_handler.Base.BaseDataset import BaseDataset
from script.data_handler.Base.BaseDatasetPack import BaseDatasetPack
from glob import glob
import numpy as np
import os
import pickle


def X_transform(x):
    x = np.reshape(x, [-1, 3, 32, 32])
    x = np_imgs_NCWH_to_NHWC(x)
    return x


def Y_transform(y):
    y = np_index_to_onehot(y)
    return y


class CIFAR10_train(BaseDataset):
    _PATTERN_TRAIN_FILE = "*/data_batch_*"
    _PKCL_KEY_TRAIN_DATA = b"data"
    _PKCL_KEY_TRAIN_LABEL = b"labels"
    LABEL_SIZE = 10

    def load(self, path):
        # load train data
        files = glob(os.path.join(path, self._PATTERN_TRAIN_FILE), recursive=True)
        files.sort()
        for file in files:
            with open(file, 'rb') as fo:
                dict_ = pickle.load(fo, encoding='bytes')

            x = dict_[self._PKCL_KEY_TRAIN_DATA]
            self.append_data('Xs', x)

            label = dict_[self._PKCL_KEY_TRAIN_LABEL]
            self.append_data('Ys', label)

    def save(self):
        pass

    def transform(self):
        self._data['Xs'] = X_transform(self._data['Xs'])
        self._data['Ys'] = Y_transform(self._data['Ys'])


class CIFAR10_test(BaseDataset):
    _PATTERN_TEST_FILE = "*/test_batch"
    _PKCL_KEY_TEST_DATA = b"data"
    _PKCL_KEY_TEST_LABEL = b"labels"
    LABEL_SIZE = 10

    def load(self, path):
        # load test data
        files = glob(os.path.join(path, self._PATTERN_TEST_FILE), recursive=True)
        files.sort()
        for file in files:
            with open(file, 'rb') as fo:
                dict_ = pickle.load(fo, encoding='bytes')

            x = dict_[self._PKCL_KEY_TEST_DATA]
            self.append_data('Xs', x)

            label = dict_[self._PKCL_KEY_TEST_LABEL]
            self.append_data('Ys', label)

    def save(self):
        pass

    def transform(self):
        self._data['Xs'] = X_transform(self._data['Xs'])
        self._data['Ys'] = Y_transform(self._data['Ys'])


class CIFAR10(BaseDatasetPack):
    def __init__(self, caching=True, **kwargs):
        super().__init__(caching, **kwargs)
        self.train_set = CIFAR10_train()
        self.test_set = CIFAR10_test()
