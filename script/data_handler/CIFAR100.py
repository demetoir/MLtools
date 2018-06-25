from script.data_handler.BaseDataset import BaseDataset, DownloadInfo
from script.data_handler.BaseDatasetPack import BaseDatasetPack
import pickle
import os
import numpy as np

BATCH_KEY_TRAIN_FINE_LABELS = "TRAIN_FINE_LABELS"
BATCH_KEY_TRAIN_COARSE_LABELS = "TRAIN_COARSE_LABELS"
BATCH_KEY_TEST_FINE_LABELS = "TEST_FINE_LABELS"
BATCH_KEY_TEST_COARSE_LABELS = "TEST_COARSE_LABELS"


def X_transform(x):
    x = np.reshape(x, [-1, 3, 32, 32])
    x = np.transpose(x, [0, 2, 3, 1])
    return x


class CIFAR100_train(BaseDataset):
    _FOLDER_NAME = 'cifar-100-python'
    _PATTERN_TRAIN_FILE = "train"
    _pkcl_key_train_data = b"data"
    _pkcl_key_train_fine_labels = b"fine_labels"
    _pkcl_key_train_coarse_labels = b"coarse_labels"

    @property
    def downloadInfos(self):
        return [
            DownloadInfo(
                url='https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
                is_zipped=True,
                download_file_name="cifar-100-python.tar.gz",
                extracted_file_names=[
                    "meta",
                    # "test",
                    "train",
                    "file.txt~"
                ]
            )
        ]

    def load(self, path, limit=None):
        # load train data
        file_path = os.path.join(path, self._FOLDER_NAME, self._PATTERN_TRAIN_FILE)
        with open(file_path, 'rb') as fo:
            dict_ = pickle.load(fo, encoding='bytes')
        x = dict_[self._pkcl_key_train_data]
        coarse_labels = dict_[self._pkcl_key_train_coarse_labels]
        fine_labels = dict_[self._pkcl_key_train_fine_labels]
        self._append_data('Xs', x)
        self._append_data(BATCH_KEY_TRAIN_COARSE_LABELS, coarse_labels)
        self._append_data(BATCH_KEY_TRAIN_FINE_LABELS, fine_labels)

    def save(self):
        pass

    def transform(self):
        self.data['Xs'] = X_transform(self.data['Xs'])


class CIFAR100_test(BaseDataset):
    _FOLDER_NAME = 'cifar-100-python'
    _PATTERN_TEST_FILE = "test"
    _pkcl_key_test_data = b"data"
    _pkcl_key_test_fine_labels = b"fine_labels"
    _pkcl_key_test_coarse_labels = b"coarse_labels"

    @property
    def downloadInfos(self):
        return [
            DownloadInfo(
                url='https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
                is_zipped=True,
                download_file_name="cifar-100-python.tar.gz",
                extracted_file_names=[
                    "meta",
                    "test",
                    # "train",
                    "file.txt~"
                ]
            )
        ]

    def load(self, path, limit=None):
        # load test data
        file_path = os.path.join(path, self._FOLDER_NAME, self._PATTERN_TEST_FILE)
        with open(file_path, 'rb') as fo:
            dict_ = pickle.load(fo, encoding='bytes')
        x = dict_[self._pkcl_key_test_data]
        coarse_labels = dict_[self._pkcl_key_test_coarse_labels]
        fine_labels = dict_[self._pkcl_key_test_fine_labels]
        self._append_data('Xs', x)
        self._append_data(BATCH_KEY_TEST_COARSE_LABELS, coarse_labels)
        self._append_data(BATCH_KEY_TEST_FINE_LABELS, fine_labels)

    def save(self):
        pass

    def transform(self):
        self.data['Xs'] = X_transform(self.data['Xs'])


class CIFAR100(BaseDatasetPack):
    def __init__(self, caching=True, **kwargs):
        super().__init__(caching, **kwargs)
        self.train_set = CIFAR100_train()
        self.test_set = CIFAR100_test()
