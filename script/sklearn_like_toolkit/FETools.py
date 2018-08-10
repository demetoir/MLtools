import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.preprocessing import Imputer, LabelEncoder

from script.util.pandas_util import df_binning

DF = pd.DataFrame


class scale_method:
    def __scale_common(self, scale, df, x_col, tail):
        return DF({
            x_col + '_' + tail: scale.fit(df[x_col])
        })

    def scale_minmax(self, df, x_col, with_scaler):
        return self.__scale_common(preprocessing.MinMaxScaler(), df, x_col, 'minmax_scale')

    def scale_max_abs(self, df, x_col):
        return self.__scale_common(preprocessing.MaxAbsScaler(), df, x_col, 'MaxAbs_scale')

    def scale_robust(self, df, x_col):
        return self.__scale_common(preprocessing.RobustScaler(), df, x_col, 'robust_scaler')

    def scale_standard(self, df, x_col):
        return self.__scale_common(preprocessing.StandardScaler(), df, x_col, 'standard_scale')

    def scale_log(self, df, x_col):
        val = np.array(list(df[x_col].astype(float)))
        return DF({
            x_col + '_log_scale': np.log(val + 1),
        })


class impute_method:
    @staticmethod
    def _imputer(df, x_col, y_col=None, missing_values="NaN", strategy="mean"):
        imputer = Imputer(missing_values=missing_values, strategy=strategy)
        imputer.fit(x_col, y_col)
        return DF({
            x_col + '_' + strategy: imputer.transform(x_col)
        })

    def impute_mean(self, df, x_col, y_col=None, missing_values='nan'):
        return self._imputer(df, x_col, y_col, missing_values, strategy='mean')

    def impute_median(self, df, x_col, y_col=None, missing_values='nan'):
        return self._imputer(df, x_col, y_col, missing_values)

    def impute_most_frequent(self, df, x_col, y_col=None, missing_value='nan'):
        return self._imputer(df, x_col, y_col, missing_value)


class encoding_method:
    def to_label(self, df, col, unique=None, with_mapper=False):
        if unique is None:
            unique = df[col].unique()

        enc = LabelEncoder()
        col_encode = col + "_label"

        enc.fit(unique)

        new_df = DF({
            col_encode: enc.transform(df[col])
        })

        if with_mapper:
            mapped = enc.transform(unique)
            encode_map = dict(zip(unique, mapped))
            decode_map = dict(zip(mapped, unique))

            return new_df, col_encode, encode_map, decode_map
        else:

            return new_df, col_encode

    def to_onehot(self, df, col, unique=None, with_mapper=False):
        if unique is None:
            unique = df[col].unique()

        n = len(df)
        d = {}
        for key in sorted(unique):
            np_arr = np.zeros(shape=[n])
            np_arr[df[col] == key] = 1
            d[col + '_onehot_' + key] = np_arr

        df = DF(d)

        return df


class clf_feature_selection:
    def varianceThreshold(self, df, col, threshold=.8):
        sel = VarianceThreshold(threshold=(threshold * (1 - threshold)))
        return sel.fit_transform(df[col])

    def kai2_select(self, df, x_col, y_col, k):
        return SelectKBest(chi2, k=k).fit_transform(df[x_col], df[y_col])

    def mutual_info_classif(self, df, x_col, y_col, k):
        raise NotImplementedError
        # return SelectKBest(chi2, k=k).fit_transform(df[x_col], df[y_col])

    def f_classif(self, df, x_col, y_col, k):
        raise NotImplementedError
        # return SelectKBest(chi2, k=k).fit_transform(df[x_col], df[y_col])


class reg_feature_selection:
    def mutual_info_regression(self):
        pass

    def f_regression(self):
        pass


class typecast_method:

    @staticmethod
    def to_str(df, key):
        df[key] = df[key].astype(str)
        return df

    @staticmethod
    def to_float(df, key):
        df[key] = df[key].astype(float)
        return df

    @staticmethod
    def to_int(df, key):
        df[key] = df[key].astype(int)
        return df


class FETools(
    encoding_method,
    impute_method,
    scale_method,
    clf_feature_selection,
    reg_feature_selection,
    typecast_method,
):

    def binning(self, df: DF, col: str, bin_seq: list, column_tail='_binning', with_intensity=False):
        return df_binning(df, col, bin_seq, column_tail, with_intensity=with_intensity)

    def update_col(self, df, old_column, new_df):
        df = df.reset_index(drop=True)
        new_df = new_df.reset_index(drop=True)

        df = df.drop(columns=old_column)
        df = pd.concat([df, new_df], axis=1)
        return df

    def concat_df(self, df_a: DF, df_b: DF):
        df_a = df_a.reset_index(drop=True)
        df_b = df_b.reset_index(drop=True)
        return pd.concat([df_a, df_b], axis=1)

    def df_group_values(self, values, new_values, df, col_key):
        for value in values:
            idxs = df.loc[:, col_key] == value
            df.loc[idxs, col_key] = new_values
        return df

    def drop_col(self, df: DF, col: str):
        return df.drop(columns=col)

    def quantile(self, df, x_col, n_quantiles=10):
        from sklearn.preprocessing import QuantileTransformer
        quantile = QuantileTransformer(n_quantiles=n_quantiles)
        quantile.fit(df[x_col])
        return quantile.transform(df[x_col])
