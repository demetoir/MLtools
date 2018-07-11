from script.data_handler.BaseDataset import BaseDataset
from script.data_handler.BaseDatasetPack import BaseDatasetPack
import pandas as pd
import numpy as np
import os
import sys

from script.data_handler.Base_df_null_handler import df_add_col_num
from script.util.misc_util import path_join

df_keys = [
    '1stFlrSF', '2ndFlrSF', '3SsnPorch', 'Alley', 'BedroomAbvGr',
    'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2',
    'BsmtFinType1', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath',
    'BsmtQual', 'BsmtUnfSF', 'CentralAir', 'Condition1', 'Condition2',
    'Electrical', 'EnclosedPorch', 'ExterCond', 'ExterQual', 'Exterior1st',
    'Exterior2nd', 'Fence', 'FireplaceQu', 'Fireplaces', 'Foundation',
    'FullBath', 'Functional', 'GarageArea', 'GarageCars', 'GarageCond',
    'GarageFinish', 'GarageQual', 'GarageType', 'GarageYrBlt', 'GrLivArea',
    'HalfBath', 'Heating', 'HeatingQC', 'HouseStyle', 'Id', 'KitchenAbvGr',
    'KitchenQual', 'LandContour', 'LandSlope', 'LotArea', 'LotConfig',
    'LotFrontage', 'LotShape', 'LowQualFinSF', 'MSSubClass', 'MSZoning',
    'MasVnrArea', 'MasVnrType', 'MiscFeature', 'MiscVal', 'MoSold',
    'Neighborhood', 'OpenPorchSF', 'OverallCond', 'OverallQual',
    'PavedDrive', 'PoolArea', 'PoolQC', 'RoofMatl', 'RoofStyle',
    'SaleCondition', 'SalePrice', 'SaleType', 'ScreenPorch', 'Street',
    'TotRmsAbvGrd', 'TotalBsmtSF', 'Utilities', 'WoodDeckSF', 'YearBuilt',
    'YearRemodAdd', 'YrSold'
]
df_Xs_keys = [
    '1stFlrSF', '2ndFlrSF', '3SsnPorch', 'Alley', 'BedroomAbvGr',
    'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2',
    'BsmtFinType1', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath',
    'BsmtQual', 'BsmtUnfSF', 'CentralAir', 'Condition1', 'Condition2',
    'Electrical', 'EnclosedPorch', 'ExterCond', 'ExterQual', 'Exterior1st',
    'Exterior2nd', 'Fence', 'FireplaceQu', 'Fireplaces', 'Foundation',
    'FullBath', 'Functional', 'GarageArea', 'GarageCars', 'GarageCond',
    'GarageFinish', 'GarageQual', 'GarageType', 'GarageYrBlt', 'GrLivArea',
    'HalfBath', 'Heating', 'HeatingQC', 'HouseStyle', 'Id', 'KitchenAbvGr',
    'KitchenQual', 'LandContour', 'LandSlope', 'LotArea', 'LotConfig',
    'LotFrontage', 'LotShape', 'LowQualFinSF', 'MSSubClass', 'MSZoning',
    'MasVnrArea', 'MasVnrType', 'MiscFeature', 'MiscVal', 'MoSold',
    'Neighborhood', 'OpenPorchSF', 'OverallCond', 'OverallQual',
    'PavedDrive', 'PoolArea', 'PoolQC', 'RoofMatl', 'RoofStyle',
    'SaleCondition', 'SaleType', 'ScreenPorch', 'Street', 'TotRmsAbvGrd',
    'TotalBsmtSF', 'Utilities', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd',
    'YrSold'
]
df_Ys_key = 'SalePrice'


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
        merged = df_add_col_num(merged, zfill_width=2)

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