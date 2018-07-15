from pprint import pprint

import pandas as pd
import numpy as np
import random
from script.data_handler.Base.Base_df_transformer import Base_df_transformer
from script.util.PlotTools import PlotTools
from script.util.numpy_utils import np_frequency_equal_bins

DF = pd.DataFrame
Series = pd.Series


def df_frequency_equal_bins(df: DF, col_key: str, n_bins: int) -> list:
    bins = np_frequency_equal_bins(np.array(df[col_key]), n_bins)
    return list(bins)


def df_value_counts(df):
    return [df[key].value_counts() for key in df]


class HousePricesTransformer(Base_df_transformer):
    def df_update_col(self, df, original_col_key, partial_df, ):
        df = df.drop(columns=original_col_key)
        df = pd.concat([df, partial_df], axis=1)
        return df

    def col_00_1stFlrSF(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        bins = df_frequency_equal_bins(partial_df, col_key, 10)

        binned_df = self.binning(df, col_key, bins)
        df = df.drop(columns=col_key)
        df = pd.concat([df, binned_df], axis=1)
        return df

    def col_01_2ndFlrSF(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # bins = df_frequency_equal_bins(partial_df, col_key, 10)
        bins = [-1, 0, 1, 423, 631, 767, 918, 2066]
        binned_df = self.binning(df, col_key, bins)

        # print(df_value_counts(binned_df))

        df = df.drop(columns=col_key)
        df = pd.concat([df, binned_df], axis=1)
        return df

    def col_02_3SsnPorch(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        df = df.drop(columns=col_key)
        return df

    def col_04_BedroomAbvGr(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        bins = [0, 2, 3, 4, 5, 8]
        binned_df = self.binning(partial_df, col_key, bins)
        # print(df_value_counts(binned_df))

        df = self.df_update_col(df, col_key, binned_df)

        return df

    def col_05_BldgType(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        df = df.drop(columns=col_key)
        return df

    def col_06_BsmtCond(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # may drop
        df[df[col_key] == 'Po'] = 'TA'
        # print(df[col_key].value_counts())
        return df

    def col_07_BsmtExposure(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # may drop
        return df

    def col_08_BsmtFinSF1(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print(partial_df.describe())
        # print(partial_df.head(20))
        # print(series.value_counts())
        # print(partial_df.info())
        #
        # print(col_key)
        # print(series)
        #
        # print(df[df[col_key] == 'TA'][col_key])
        # print(df[df[col_key] == 'TA'][col_key])

        df.loc[df[col_key] == 'TA', col_key] = 0.0
        df[col_key] = df[col_key].astype(float)
        partial_df = df[[col_key]]
        bins = [-1.0, 0.0, 1, 196.0, 368.0, 512.0, 654.0, 808.0, 1047.0, 5645.0]
        binned_df = self.binning(partial_df, col_key, bins)

        # print(binned_df.info())
        # print(df_value_counts(binned_df))

        # plot = PlotTools(save=False, show=True)
        # plot.count(binned_df, col_key)
        # plot.joint_2d(binned_df, col_key, Ys_key)

        self.df_update_col(df, col_key, binned_df)
        return df

    def col_09_BsmtFinSF2(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        df.loc[df[col_key] == 'TA', col_key] = 0
        partial_df = df[[col_key]]

        partial_df = partial_df.astype('float')
        # print(partial_df.info())

        bins = df_frequency_equal_bins(partial_df, col_key, 10)
        # print(bins)
        binned_df = self.binning(partial_df, col_key, bins)
        # print(df_value_counts(binned_df))

        df = df.drop(columns=col_key)

        return df

    def col_10_BsmtFinType1(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        print(partial_df.info())
        print(df_value_counts(partial_df))
        return df

    def col_11_BsmtFinType2(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_12_BsmtFullBath(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_13_BsmtHalfBath(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_14_BsmtQual(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_15_BsmtUnfSF(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_16_CentralAir(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_17_Condition1(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_18_Condition2(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_19_Electrical(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_20_EnclosedPorch(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list,
                             Ys_key: list):
        return df

    def col_21_ExterCond(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_22_ExterQual(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_23_Exterior1st(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_24_Exterior2nd(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_26_FireplaceQu(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_27_Fireplaces(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_28_Foundation(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_29_FullBath(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_30_Functional(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_31_GarageArea(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_32_GarageCars(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_33_GarageCond(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_34_GarageFinish(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_35_GarageQual(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_36_GarageType(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_37_GarageYrBlt(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_38_GrLivArea(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_39_HalfBath(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_40_Heating(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_41_HeatingQC(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_42_HouseStyle(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_43_Id(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        df = df.drop(columns=col_key)
        return df

    def col_44_KitchenAbvGr(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_45_KitchenQual(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_46_LandContour(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_47_LandSlope(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_48_LotArea(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_49_LotConfig(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_50_LotFrontage(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_51_LotShape(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_52_LowQualFinSF(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_53_MSSubClass(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_54_MSZoning(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_55_MasVnrArea(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_56_MasVnrType(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_58_MiscVal(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_59_MoSold(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_60_Neighborhood(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_61_OpenPorchSF(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_62_OverallCond(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_63_OverallQual(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_64_PavedDrive(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_65_PoolArea(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_67_RoofMatl(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_68_RoofStyle(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_69_SaleCondition(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list,
                             Ys_key: list):
        return df

    def col_70_SalePrice(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_71_SaleType(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_72_ScreenPorch(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_73_Street(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_74_TotRmsAbvGrd(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_75_TotalBsmtSF(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_77_WoodDeckSF(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_78_YearBuilt(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_79_YearRemodAdd(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_80_YrSold(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df
