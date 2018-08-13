from tensorflow.examples.tutorials.mnist import input_data
from script.data_handler.Base.BaseDataset import BaseDataset, DownloadInfo
from script.data_handler.Base.BaseDatasetPack import BaseDatasetPack
import numpy as np


def X_transform(x):
    shape = x.shape
    x = np.reshape(x, [shape[0], 28, 28])
    npad = ((0, 0), (2, 2), (2, 2))
    x = np.pad(x, pad_width=npad, mode='constant', constant_values=0)
    x = np.reshape(x, [shape[0], 32, 32, 1])
    return x


class Fashion_MNIST_train(BaseDataset):
    TRAIN_SIZE = 60000
    LABEL_SIZE = 10

    @property
    def downloadInfos(self):
        return [
            DownloadInfo(
                url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/',
                is_zipped=True,
                download_file_name='...',
                extracted_file_names=[]
            )
        ]

    def load(self, path, limit=None):
        fashion_mnist = input_data.read_data_sets(path,
                                                  source_url=self.downloadInfos[0].url,
                                                  one_hot=True)
        # load train data
        Xs, Ys = fashion_mnist.train.next_batch(self.TRAIN_SIZE)
        self._data['Xs'] = Xs
        self._data['Ys'] = Ys

    def save(self):
        pass

    def transform(self):
        self._data['Xs'] = X_transform(self._data['Xs'])


class Fashion_MNIST_test(BaseDataset):
    TEST_SIZE = 10000
    LABEL_SIZE = 10
    download_infos = [
        DownloadInfo(
            url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/',
            is_zipped=True,
            download_file_name='...',
            extracted_file_names=[]
        )
    ]

    def load(self, path, limit=None):
        fashion_mnist = input_data.read_data_sets(path,
                                                  source_url=self.download_infos[0].url,
                                                  one_hot=True)
        Xs, Ys = fashion_mnist.test.next_batch(self.TEST_SIZE)
        self._data['Xs'] = Xs
        self._data['Ys'] = Ys

    def save(self):
        pass

    def transform(self):
        self._data['Xs'] = X_transform(self._data['Xs'])


class Fashion_MNIST(BaseDatasetPack):
    def __init__(self, caching=True, **kwargs):
        super().__init__(caching, **kwargs)
        self.train_set = Fashion_MNIST_train()
        self.test_set = Fashion_MNIST_test()
