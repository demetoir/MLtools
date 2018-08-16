import math
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.preprocessing import Imputer, LabelEncoder
from script.util.MixIn import PickleMixIn
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


class DF_binning_encoder(PickleMixIn):
    def __init__(self):
        PickleMixIn.__init__(self)

        self.is_fit = False

        self.decode_mapper = None
        self.bins = None
        self.col = None
        self.encode_uniques = None
        self.encode_method = None
        self.decode_method = None

    def check_fit(self):
        if not self.is_fit:
            raise ValueError(f'{self.__class__.__name__} has not fitted')

    def make_decode_mapper(self, df, col):
        self.decode_mapper = {}
        bins = self.bins
        for i in range(len(bins) - 1):
            a, b = bins[i: i + 2]
            encode_value = f'bin{str(i).zfill(int(math.log10(len(bins))) + 1)}_[{a}~{b})'

            query = f'{a} <= {col} < {b}'
            idx = list(df.query(query).index.values)
            value_count = df[idx, col].value_counts()

            if self.decode_method == 'major':
                major_value = value_count[0]
                self.decode_mapper[encode_value] = major_value
            else:
                raise ValueError(f'{self.decode_method} does not support')

    def _major_method(self, df, col, bins):
        value_count = df[col].value_counts()
        vc_df = DF({'count': value_count.values, col: value_count.index})
        vc_df = pd.concat([vc_df, df_binning(vc_df, col, bins)], axis=1)

        mapper = {unique: (-1, None) for unique in self.encode_uniques}
        for idx in range(len(vc_df)):
            encode_value = str(vc_df.loc[idx, col + '_binning'])
            count = int(vc_df.loc[idx, 'count'])
            col_value = float(vc_df.loc[idx, col])

            if count > mapper[encode_value][0]:
                mapper[encode_value] = (count, col_value)

        mapper = {key: val[1] for key, val in mapper.items()}
        return mapper

    def fit(self, df, col, bins, encode_method=None, decode_method='major'):
        self.col = col
        self.bins = bins
        self.encode_method = encode_method
        self.decode_method = decode_method

        self.encode_uniques = []
        for i in range(len(bins) - 1):
            a, b = bins[i: i + 2]
            encode_value = f'bin{str(i).zfill(int(math.log10(len(bins))) + 1)}_[{a}~{b})'
            self.encode_uniques += [encode_value]

        self.decode_mapper = {}
        if self.decode_method == 'major':
            self.decode_mapper = self._major_method(df, col, bins)
        else:
            raise ValueError(f'{self.decode_method} does not support')

        self.is_fit = True

    def encode(self, df):
        self.check_fit()

        df_encode = df_binning(df[[self.col]], self.col, self.bins, column_tail='')
        return df_encode

    def decode(self, df):
        self.check_fit()

        df_decoded = DF({self.col: np.zeros([len(df)])})
        unique = df[self.col].unique()
        for value in unique:
            if value not in self.encode_uniques:
                raise ValueError(f'{value} can not decode')

            idxs = df[self.col] == value
            df_decoded.loc[idxs, self.col] = self.decode_mapper[value]

        return df_decoded

    def encode_to_np(self, df):
        return self.to_np(self.encode(df))

    def decode_from_np(self, np_arr):
        return self.decode(self.from_np(np_arr))

    def to_np(self, df: DF):
        ret = {}
        for key in df.keys():
            np_arr = np.array(df[key].values)
            np_arr = np_arr.reshape([len(np_arr), 1])
            ret[key] = np_arr

        return np.concatenate([v for k, v in ret.items()], axis=1)

    def from_np(self, np_arr, np_cols=None):
        if np_cols is None:
            np_cols = np_arr.reshape([len(np_arr), -1]).shape[1]

        df = DF()
        for idx, col in enumerate(np_cols):
            df[col] = np_arr[:, idx]

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

    def drop_row_by_value(self, df: DF, col, value):
        return df.loc[df[col] != value, :]

    def quantile(self, df, x_col, n_quantiles=10):
        from sklearn.preprocessing import QuantileTransformer
        quantile = QuantileTransformer(n_quantiles=n_quantiles)
        quantile.fit(df[x_col])
        return quantile.transform(df[x_col])
