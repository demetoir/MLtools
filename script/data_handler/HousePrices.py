from script.data_handler.Base.BaseDataset import BaseDataset
from script.data_handler.Base.BaseDatasetPack import BaseDatasetPack
import pandas as pd
import os

from script.data_handler.HousePricesCleaner import HousePricesCleaner
from script.data_handler.HousePricesTransformer import HousePricesTransformer
from script.util.misc_util import path_join

df_raw_keys = [
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
    'col_00_1stFlrSF', 'col_01_2ndFlrSF', 'col_02_3SsnPorch',
    'col_04_BedroomAbvGr', 'col_05_BldgType', 'col_06_BsmtCond',
    'col_07_BsmtExposure', 'col_08_BsmtFinSF1', 'col_09_BsmtFinSF2',
    'col_10_BsmtFinType1', 'col_11_BsmtFinType2', 'col_12_BsmtFullBath',
    'col_13_BsmtHalfBath', 'col_14_BsmtQual', 'col_15_BsmtUnfSF',
    'col_16_CentralAir', 'col_17_Condition1', 'col_18_Condition2',
    'col_19_Electrical', 'col_20_EnclosedPorch', 'col_21_ExterCond',
    'col_22_ExterQual', 'col_23_Exterior1st', 'col_24_Exterior2nd',
    'col_26_FireplaceQu', 'col_27_Fireplaces', 'col_28_Foundation',
    'col_29_FullBath', 'col_30_Functional', 'col_31_GarageArea',
    'col_32_GarageCars', 'col_33_GarageCond', 'col_34_GarageFinish',
    'col_35_GarageQual', 'col_36_GarageType', 'col_37_GarageYrBlt',
    'col_38_GrLivArea', 'col_39_HalfBath', 'col_40_Heating',
    'col_41_HeatingQC', 'col_42_HouseStyle', 'col_43_Id',
    'col_44_KitchenAbvGr', 'col_45_KitchenQual', 'col_46_LandContour',
    'col_47_LandSlope', 'col_48_LotArea', 'col_49_LotConfig',
    'col_50_LotFrontage', 'col_51_LotShape', 'col_52_LowQualFinSF',
    'col_53_MSSubClass', 'col_54_MSZoning', 'col_55_MasVnrArea',
    'col_56_MasVnrType', 'col_58_MiscVal', 'col_59_MoSold',
    'col_60_Neighborhood', 'col_61_OpenPorchSF', 'col_62_OverallCond',
    'col_63_OverallQual', 'col_64_PavedDrive', 'col_65_PoolArea',
    'col_67_RoofMatl', 'col_68_RoofStyle', 'col_69_SaleCondition',
    'col_71_SaleType', 'col_72_ScreenPorch',
    'col_73_Street', 'col_74_TotRmsAbvGrd', 'col_75_TotalBsmtSF',
    'col_77_WoodDeckSF', 'col_78_YearBuilt', 'col_79_YearRemodAdd',
    'col_80_YrSold'
]
df_Ys_key = 'col_70_SalePrice'

DF = pd.DataFrame


class HousePricesHelper:
    @staticmethod
    def null_cleaning(merge_df):
        nullCleaner = HousePricesCleaner(merge_df, df_Xs_keys, 'col_70_SalePrice', silent=True)
        # info = nullCleaner.null_cols_info()
        # print(info)
        # nullCleaner.null_cols_plot()
        nullCleaner.boilerplate_maker(path='./gen_code.py')

        merge_null_clean = nullCleaner.clean()
        # print(merge_null_clean.info())
        return merge_null_clean

    @staticmethod
    def transform(merge_df: DF) -> DF:
        transformer = HousePricesTransformer(merge_df, df_Xs_keys, df_Ys_key)
        transformer.boilerplate_maker('./gen_code.py')
        transformer.plot_all()

        df = transformer.transform()
        transformer.corr_heatmap()

        # transformer = HousePricesTransformer(merge_df[['col_00_1stFlrSF', df_Ys_key]], df_Xs_keys, df_Ys_key)

        return df

    @staticmethod
    def train_test_split(merged_df: DF) -> (DF, DF):
        test = merged_df.query(f'{df_Ys_key}.isnull()')
        test = test.drop(columns=[df_Ys_key])

        train = merged_df.query(f'not {df_Ys_key}.isnull()')

        return train, test

    @staticmethod
    def load_merge_set(path):
        def df_add_col_num(df, zfill_width=None):
            if zfill_width is None:
                zfill_width = 0

            mapping = {}
            for idx, key in enumerate(df.keys()):
                mapping[key] = f'col_{str(idx).zfill(zfill_width)}_{key}'

            return df.rename(mapping, axis='columns')

        merged_path = path_join(path, 'merged.csv')
        if os.path.exists(merged_path):
            merged = pd.read_csv(merged_path)
        else:
            train_path = path_join(path, 'train.csv')
            train = pd.read_csv(train_path)

            test_path = path_join(path, 'test.csv')
            test = pd.read_csv(test_path)

            merged = pd.concat([train, test], axis=0)
            merged = df_add_col_num(merged, zfill_width=2)

            merged.to_csv(merged_path, index=False)
        return merged


class HousePrices_train(BaseDataset):

    def load(self, path, limit=None):
        train_path = path_join(path, 'transformed_train.csv')

        merged_df = HousePricesHelper.load_merge_set(path)

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
