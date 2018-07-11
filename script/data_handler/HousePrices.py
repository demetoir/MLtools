from script.data_handler.BaseDataset import BaseDataset
from script.data_handler.BaseDatasetPack import BaseDatasetPack
import pandas as pd
import numpy as np
import os
import sys

from script.util.misc_util import path_join


def df_add_col_num(df):
    ret_df = None

    return ret_df


def load_merge_set(path):
    merged_path = path_join(path, 'merged.csv')
    if os.path.exists(merged_path):
        merged = pd.read_csv(merged_path)
    else:
        train_path = path_join(path, 'train.csv')
        test_path = path_join(path, 'test.csv')

        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        merged = pd.concat([train, test], axis=0)
        merged.to_csv(merged_path, index=False)
    return merged


class HousePrices_train(BaseDataset):

    def load(self, path, limit=None):
        train_path = path_join(path, 'transformed_train.csv')

        merged_df = load_merge_set(path)

        pass


class HousePrices_test(BaseDataset):
    def load(self, path, limit=None):
        pass


class HousePrices(BaseDatasetPack):
    def __init__(self, caching=True, verbose=0, **kwargs):
        super().__init__(caching, verbose, **kwargs)
        self.pack['train'] = HousePrices_train()
        self.pack['test'] = HousePrices_test()

    def to_kaggle_submit_csv(self, predict):
        raise NotImplementedError
