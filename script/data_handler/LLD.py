from __future__ import division
from glob import glob
from script.data_handler.Base.BaseDataset import BaseDataset
from script.data_handler.Base.BaseDatasetPack import BaseDatasetPack
import _pickle as pickle
import os


class LLD_clean(BaseDataset):
    LLD_CLEAN = 'CLEAN'
    LLD_FULL = 'FULL'
    PATTERN = 'LLD_favicon_data*.pkl'

    def load(self, path):
        files = glob(os.path.join(path, self.PATTERN))
        files.sort()
        self._data['x'] = None
        for file in files:
            with open(file, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                self.log('pickle load :%s' % file)
            self.append_data('x', data)

    def save(self):
        # def save_icon_data(icons, data_path, package_size=100000):
        #     if not os.instance_path.exists(data_path):
        #         os.makedirs(data_path)
        #     num_packages = int(math.ceil(len(icons) / package_size))
        #     num_len = len(str(num_packages))
        #     for p in range(num_packages):
        #         with open(os.instance_path.join(data_path, 'icon_data_' + str(p).zfill(num_len) + '.pkl'), 'wb') as f:
        #             cPickle.dump(icons[p * package_size:(p + 1) * package_size], f, protocol=cPickle.HIGHEST_PROTOCOL)
        pass

    def transform(self):
        pass


class LLD(BaseDatasetPack):

    def __init__(self, caching=True, **kwargs):
        super().__init__(caching, **kwargs)
        self.train_set = LLD_clean()
