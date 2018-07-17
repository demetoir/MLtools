import numpy as np
import pandas as pd
from script.data_handler.Base.Base_df_transformer import Base_df_transformer
from script.util.numpy_utils import np_frequency_equal_bins

DF = pd.DataFrame
Series = pd.Series


def df_frequency_equal_bins(df: DF, col_key: str, n_bins: int) -> list:
    bins = np_frequency_equal_bins(np.array(df[col_key]), n_bins)
    return list(bins)


def df_value_counts(df):
    return [df[key].value_counts() for key in df]


def print_info(df, col_key, partial_df, series, Xs_keys, Ys_key):
    print(col_key)
    print(partial_df.info())
    print(df_value_counts(partial_df))
    print()


class HousePricesTransformer(Base_df_transformer):

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
        # df[df[col_key] == 'Po'] = 'TA'
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

        # df.loc[df[col_key] == 'TA', col_key] = 0.0
        # df[col_key] = df[col_key].astype(float)
        partial_df = df[[col_key]]
        bins = [-1.0, 0.0, 1, 196.0, 368.0, 512.0, 654.0, 808.0, 1047.0, 5645.0]
        binned_df = self.binning(partial_df, col_key, bins)

        # print(binned_df.info())
        # print(df_value_counts(binned_df))

        # plot = PlotTools(save=False, show=True)
        # plot.count(binned_df, col_key)
        # plot.joint_2d(binned_df, col_key, Ys_key)

        df = self.df_update_col(df, col_key, binned_df)
        return df

    def col_09_BsmtFinSF2(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # bins = df_frequency_equal_bins(partial_df, col_key, 10)
        # print(bins)
        # binned_df = self.binning(partial_df, col_key, bins)
        # print(df_value_counts(binned_df))

        df = df.drop(columns=col_key)

        return df

    def col_10_BsmtFinType1(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print(partial_df.info())
        # print(df_value_counts(partial_df))
        return df

    def col_11_BsmtFinType2(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)
        return df

    def col_12_BsmtFullBath(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)
        return df

    def col_13_BsmtHalfBath(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)
        return df

    def col_14_BsmtQual(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)
        return df

    def col_15_BsmtUnfSF(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # todo may better binning
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)
        bins = df_frequency_equal_bins(partial_df, col_key, 10)
        # print(bins)
        binned_df = self.binning(partial_df, col_key, bins)
        # print(df_value_counts(binned_df))

        df = self.df_update_col(df, col_key, binned_df)

        return df

    def col_16_CentralAir(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)
        return df

    def col_17_Condition1(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)
        return df

    def col_18_Condition2(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)

        return df

    def col_19_Electrical(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)

        return df

    def col_20_EnclosedPorch(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # TODO better binning
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)
        # bins = df_frequency_equal_bins(partial_df, col_key, 10)
        bins = [-1.0, 0.0, 1.0, 100.0, 200.0, 300.0, 400.0, 1013.0]
        # print(bins)
        binned_df = self.binning(partial_df, col_key, bins)
        # print(df_value_counts(binned_df))

        df = self.df_update_col(df, col_key, binned_df)

        return df

    def col_21_ExterCond(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)

        return df

    def col_22_ExterQual(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)

        return df

    def col_23_Exterior1st(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)
        return df

    def col_24_Exterior2nd(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)
        return df

    def col_26_FireplaceQu(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)
        return df

    def col_27_Fireplaces(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)

        idxs = df.loc[:, col_key] == '2'
        df.loc[idxs, col_key] = '2~4'
        idxs = df.loc[:, col_key] == '3'
        df.loc[idxs, col_key] = '2~4'
        idxs = df.loc[:, col_key] == '4'
        df.loc[idxs, col_key] = '2~4'
        # print(df_value_counts(df[[col_key]]))

        return df

    def col_28_Foundation(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)

        idxs = df.loc[:, col_key] == 'Stone'
        df.loc[idxs, col_key] = 'Stone_and_Wood'

        idxs = df.loc[:, col_key] == 'Wood'
        df.loc[idxs, col_key] = 'Stone_and_Wood'

        # print(df_value_counts(df[[col_key]]))

        return df

    def col_29_FullBath(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        df[[col_key]] = df[[col_key]].astype(str)

        idxs = df.loc[:, col_key] == '4'
        df.loc[idxs, col_key] = '3~4'
        idxs = df.loc[:, col_key] == '3'
        df.loc[idxs, col_key] = '3~4'

        idxs = df.loc[:, col_key] == '0'
        df.loc[idxs, col_key] = '0~1'
        idxs = df.loc[:, col_key] == '1'
        df.loc[idxs, col_key] = '0~1'

        # print(df_value_counts(df[[col_key]]))

        # print_info(df[[col_key]], col_key, partial_df, series, Xs_key, Ys_key)

        return df

    def col_30_Functional(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        df = df.drop(columns=col_key)
        return df

    def col_31_GarageArea(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)

        bins = df_frequency_equal_bins(df, col_key, 10)
        # print(bins)

        binning = self.binning(df, col_key, bins)
        # print(df_value_counts(binning))

        df = self.df_update_col(df, col_key, binning)

        return df

    def col_32_GarageCars(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)

        idxs = df[col_key] == '5.0'
        df.loc[idxs, col_key] = '4~5'

        idxs = df[col_key] == '4.0'
        df.loc[idxs, col_key] = '4~5'
        # print(df_value_counts(df[[col_key]]))

        return df

    def col_33_GarageCond(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        df = df.drop(columns=col_key)
        return df

    def col_34_GarageFinish(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_35_GarageQual(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        df = df.drop(columns=col_key)
        return df

    def col_36_GarageType(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print_info(df, col_key, partial_df,series, Xs_key, Ys_key)

        idxs = df[col_key] == 'Basment'
        df.loc[idxs, col_key] = 'etc'
        idxs = df[col_key] == '2Types'
        df.loc[idxs, col_key] = 'etc'
        idxs = df[col_key] == 'CarPort'
        df.loc[idxs, col_key] = 'etc'

        # print(df_value_counts(df[[col_key]]))

        return df

    def col_37_GarageYrBlt(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # TODO better binning
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)
        bins = df_frequency_equal_bins(partial_df, col_key, 10)
        # print(bins)
        binned_df = self.binning(partial_df, col_key, bins)
        # print(df_value_counts(binned_df))

        df = self.df_update_col(df, col_key, binned_df)
        return df

    def col_38_GrLivArea(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # TODO better binning
        bins = df_frequency_equal_bins(partial_df, col_key, 10)
        binned_df = self.binning(partial_df, col_key, bins)
        # print(bins)
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)
        # print(df_value_counts(binned_df))

        df = self.df_update_col(df, col_key, binned_df)
        return df

    def col_39_HalfBath(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_40_Heating(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        df = df.drop(columns=col_key)
        return df

    def col_41_HeatingQC(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)
        idxs = df.loc[:, col_key] == 'Po'
        df.loc[idxs, col_key] = 'TA'
        # print(df_value_counts(df[[col_key]]))
        return df

    def col_42_HouseStyle(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)

        idxs = df.loc[:, col_key] == 'SFoyer'
        df.loc[idxs, col_key] = 'etc'

        idxs = df.loc[:, col_key] == '2.5Unf'
        df.loc[idxs, col_key] = 'etc'

        idxs = df.loc[:, col_key] == '1.5Unf'
        df.loc[idxs, col_key] = 'etc'

        idxs = df.loc[:, col_key] == '2.5Fin'
        df.loc[idxs, col_key] = 'etc'
        # print(df_value_counts(df[[col_key]]))

        return df

    def col_43_Id(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        df = df.drop(columns=col_key)
        return df

    def col_44_KitchenAbvGr(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        df = df.drop(columns=col_key)
        return df

    def col_45_KitchenQual(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_46_LandContour(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)
        return df

    def col_47_LandSlope(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        df = df.drop(columns=col_key)
        return df

    def col_48_LotArea(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        bins = df_frequency_equal_bins(partial_df, col_key, 10)
        binned_df = self.binning(partial_df, col_key, bins)
        # print(bins)
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)
        # print(df_value_counts(binned_df))

        df = self.df_update_col(df, col_key, binned_df)
        return df

    def col_49_LotConfig(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        df = df.drop(columns=col_key)
        return df

    def col_50_LotFrontage(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        bins = df_frequency_equal_bins(partial_df, col_key, 10)
        binned_df = self.binning(partial_df, col_key, bins)
        # print(bins)
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)
        # print(df_value_counts(binned_df))

        df = self.df_update_col(df, col_key, binned_df)
        return df

    def col_51_LotShape(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        idxs = df.loc[:, col_key] == 'IR2'
        df.loc[idxs, col_key] = 'IR12'

        idxs = df.loc[:, col_key] == 'IR3'
        df.loc[idxs, col_key] = 'IR23'
        # print(df_value_counts(df[[col_key]]))
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)

        return df

    def col_52_LowQualFinSF(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        df = df.drop(columns=col_key)
        return df

    def col_53_MSSubClass(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)

        df = self.df_group_values([30, 40, 45], 40, df, col_key)
        df = self.df_group_values([150, 160, 180, 190], 155, df, col_key)
        # print(df_value_counts(df[[col_key]]))

        return df

    def col_54_MSZoning(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_55_MasVnrArea(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # bins = df_frequency_equal_bins(partial_df, col_key, 15)
        bins = [-1.0, 0.0, 85.0, 144.0, 200.0, 270.0, 408.0, 1601.0]
        binned_df = self.binning(partial_df, col_key, bins)
        # print(bins)
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)
        # print(df_value_counts(binned_df))

        df = self.df_update_col(df, col_key, binned_df)
        return df

    def col_56_MasVnrType(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_58_MiscVal(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        df = df.drop(columns=col_key)
        return df

    def col_59_MoSold(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_60_Neighborhood(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_61_OpenPorchSF(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # bins = df_frequency_equal_bins(partial_df, col_key, 15)
        bins = [-1, 0, 20, 32, 40, 50, 64, 84, 112, 160, 743]
        binned_df = self.binning(partial_df, col_key, bins)
        # print(bins)
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)
        # print(df_value_counts(binned_df))

        df = self.df_update_col(df, col_key, binned_df)

        return df

    def col_62_OverallCond(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        df = self.df_group_values(['1', '2', '3'], '1~3', df, col_key)
        # print(df_value_counts(df[[col_key]]))
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)
        return df

    def col_63_OverallQual(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)
        df = self.df_group_values(['1', '2', '3'], '1~3', df, col_key)
        # print(df_value_counts(df[[col_key]]))
        return df

    def col_64_PavedDrive(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_65_PoolArea(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        df = df.drop(columns=col_key)
        return df

    def col_67_RoofMatl(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        df = df.drop(columns=col_key)
        return df

    def col_68_RoofStyle(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_69_SaleCondition(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list,
                             Ys_key: list):
        df = df.drop(columns=col_key)
        return df

    def col_71_SaleType(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)
        values = ['ConLD', 'CWD', 'ConLI', 'ConLw', 'Oth', 'Con', 'COD']
        df = self.df_group_values(values, 'etc', df, col_key)
        # print(df_value_counts(df[[col_key]]))

        return df

    def col_72_ScreenPorch(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)
        #
        # [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 577]
        # bins = df_frequency_equal_bins(df, col_key, 10)
        # print(bins)
        # binning_df = self.binning(df, col_key, bins)
        # print(df_value_counts(binning_df))

        df = df.drop(columns=col_key)
        return df

    def col_73_Street(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        df = df.drop(columns=col_key)
        return df

    def col_74_TotRmsAbvGrd(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)
        df = self.df_group_values(['2', '3'], '2~3', df, col_key)
        df = self.df_group_values(['11', '12', '13', '14', '15'], '11~15', df, col_key)

        # print(df_value_counts(df[[col_key]]))
        return df

    def col_75_TotalBsmtSF(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)
        bins = df_frequency_equal_bins(df, col_key, 10)
        # print(bins)

        binning_df = self.binning(df, col_key, bins)
        df = self.df_update_col(df, col_key, binning_df)
        # print(df_value_counts(binning_df))
        # print(df.info())
        return df

    def col_77_WoodDeckSF(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)
        # bins = df_frequency_equal_bins(df, col_key, 10)
        bins = [-1, 0, 1, 100, 144, 192, 256, 1425]
        # print(bins)

        binning_df = self.binning(df, col_key, bins)
        df = self.df_update_col(df, col_key, binning_df)
        # print(df_value_counts(binning_df))
        return df

    def col_78_YearBuilt(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print_info(df, col_key, partial_df, series, Xs_key, Ys_key)
        bins = df_frequency_equal_bins(df, col_key, 20)
        # print(bins)

        binning_df = self.binning(df, col_key, bins)
        df = self.df_update_col(df, col_key, binning_df)
        # print(df_value_counts(binning_df))
        return df

    def col_79_YearRemodAdd(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        print_info(df, col_key, partial_df, series, Xs_key, Ys_key)
        bins = df_frequency_equal_bins(df, col_key, 10)
        # print(bins)

        binning_df = self.binning(df, col_key, bins)
        df = self.df_update_col(df, col_key, binning_df)
        # print(df_value_counts(binning_df))
        return df

    def col_80_YrSold(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        df = df.drop(columns=col_key)
        return df
