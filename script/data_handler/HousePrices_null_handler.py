import pandas as pd
import numpy as np
import random
from script.data_handler.Base_df_null_handler import Base_df_null_handler

DF = pd.DataFrame
Series = pd.Series


class HousePrices_null_handler(Base_df_null_handler):

    def col_03_Alley(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        df = df.drop(columns=key)

        return df

    def col_06_BsmtCond(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        df = self.fill_major_value_cate(df, key)

        return df

    def col_07_BsmtExposure(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        df = self.fill_major_value_cate(df, key)

        return df

    def col_08_BsmtFinSF1(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_09_BsmtFinSF2(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_10_BsmtFinType1(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_11_BsmtFinType2(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_12_BsmtFullBath(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_13_BsmtHalfBath(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_14_BsmtQual(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        self.print_null_col_info(df, key)

        return df

    def col_15_BsmtUnfSF(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_19_Electrical(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_23_Exterior1st(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_24_Exterior2nd(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_25_Fence(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        df = df.drop(columns=key)

        return df

    def col_26_FireplaceQu(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_30_Functional(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_31_GarageArea(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_32_GarageCars(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_33_GarageCond(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_34_GarageFinish(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_35_GarageQual(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_36_GarageType(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_37_GarageYrBlt(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_45_KitchenQual(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_50_LotFrontage(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_54_MSZoning(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_55_MasVnrArea(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_56_MasVnrType(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_57_MiscFeature(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        df = df.drop(columns=key)

        return df

    def col_66_PoolQC(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        df = df.drop(columns=key)

        return df

    def col_75_TotalBsmtSF(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_76_Utilities(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df
