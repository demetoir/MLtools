from sklearn import preprocessing
from pandas import DataFrame as DF
from pandas import Series
import pandas as pd
import numpy as np
from script.util.MixIn import PickleMixIn


class DF_encoder(PickleMixIn):
    scale_method = {
        'minmax': preprocessing.MinMaxScaler,
        'maxabs': preprocessing.MaxAbsScaler,
        'robust': preprocessing.RobustScaler,
        'standard': preprocessing.StandardScaler,
    }

    def __init__(self):
        self.cate_cols = []
        self.onehot_uniques = {}
        self.onehot_cols = {}
        self.onehot_df_cols_full = []

        self.scalers = {}
        self.conti_cols = []
        self.scaled_cols_full = []
        self.method = None

        self.cols = []
        self.encoded_cols = []

        self.np_cols = []
        self.np_dtypes = []

    def encode_cate(self, df: DF):
        n = len(df)

        onehot_df = DF()
        for col in df.columns:
            uniques = sorted(list(df[col].unique()))
            self.onehot_uniques[col] = uniques

            onehot_cols = [col + '_onehot_' + unique_val for unique_val in uniques]
            self.onehot_cols[col] = onehot_cols

            for unique_val in uniques:
                np_arr = np.zeros(shape=[n])
                np_arr[df[col] == unique_val] = 1
                onehot_df[col + '_onehot_' + unique_val] = np_arr

        self.onehot_df_cols_full = list(onehot_df.columns)

        return onehot_df

    def encode_conti(self, df: DF, method=None):
        self.method = method
        if self.method is None:
            self.scaled_cols_full = self.conti_cols
            return df

        scaled_df = DF()
        for col in df.columns:
            scale_method = self.scale_method[method]
            scaler = scale_method()

            np_arr = np.array(df[col]).reshape([-1, 1])
            scaler.fit(np_arr)
            np_arr = scaler.transform(np_arr)
            scaled_df[col + f'_{method}_scaled'] = np_arr.reshape([-1])

            self.scalers[col] = scaler

        self.scaled_cols_full = list(scaled_df.columns)

        return scaled_df

    def encode(self, df, cate_cols, conti_cols, scale_method=None):
        self.cols = cate_cols + conti_cols
        self.cate_cols = cate_cols
        self.conti_cols = conti_cols

        cate_df = df[cate_cols]
        conti_df = df[conti_cols]

        cate_df_encoded = self.encode_cate(cate_df)
        conti_df_encoded = self.encode_conti(conti_df, method=scale_method)

        concat_df = pd.concat([cate_df_encoded, conti_df_encoded], axis=1)
        concat_df = concat_df[sorted(list(concat_df.columns))]
        self.encoded_cols = list(concat_df.columns)

        return concat_df

    @staticmethod
    def _decode_onehot(df, uniques):
        n = len(df)
        a = Series(np.zeros(shape=[n]))

        for col, unique in zip(list(df.columns), uniques):
            a[df[df[col] == 1].index] = unique

        return a

    def decode_cate(self, df: DF):
        decoded_df = DF()
        for col in self.cate_cols:
            uniques = self.onehot_uniques[col]
            onehot_cols = self.onehot_cols[col]
            decoded_df[col] = self._decode_onehot(df[onehot_cols], uniques)

        return decoded_df

    def decode_conti(self, df: DF):
        if self.method is None:
            return df

        decoded_df = DF()
        for scaled_col, conti_col in zip(self.scaled_cols_full, self.conti_cols):
            scaler = self.scalers[conti_col]

            np_arr = np.array(df[scaled_col]).reshape([-1, 1])
            np_arr = scaler.inverse_transform(np_arr)
            decoded_df[conti_col] = np_arr.reshape([-1])

        return decoded_df

    def decode(self, df: DF):
        onehot_df = df[self.onehot_df_cols_full]
        cate_df = self.decode_cate(onehot_df)

        scaled_df = df[self.scaled_cols_full]
        conti_df = self.decode_conti(scaled_df)

        decoded_df = pd.concat([cate_df, conti_df], axis=1)
        decoded_df = decoded_df[self.cols]

        return decoded_df

    def to_np(self, df: DF):
        self.np_cols = list(df.columns)
        self.np_dtypes = [df[key].dtype for key in df.keys()]

        ret = {}
        for key, dtype in zip(df.keys(), self.np_dtypes):
            np_arr = np.array(df[key].values, dtype=dtype)
            np_arr = np_arr.reshape([len(np_arr), 1])
            ret[key] = np_arr

        return np.concatenate([v for k, v in ret.items()], axis=1)

    def from_np(self, np_arr, np_cols=None):
        if np_cols is None:
            np_cols = self.np_cols

        df = DF()
        for idx, col in enumerate(np_cols):
            df[col] = np_arr[:, idx]

        return df
