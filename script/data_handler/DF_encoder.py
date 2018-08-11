from sklearn import preprocessing
from pandas import DataFrame as DF
from pandas import Series
import pandas as pd
import numpy as np
from script.util.MixIn import PickleMixIn

np_number_types = (
    np.int, np.int8, np.int16, np.int32, np.int64,
    np.float, np.float16, np.float32, np.float64,
    np.uint, np.uint8, np.uint16, np.uint32, np.uint64
)


class DF_encoder(PickleMixIn):
    scale_method_class = {
        'minmax': preprocessing.MinMaxScaler,
        'maxabs': preprocessing.MaxAbsScaler,
        'robust': preprocessing.RobustScaler,
        'standard': preprocessing.StandardScaler,
    }

    def __init__(self):
        self.cate_cols = []
        self.cate_encoded_cols = []
        self.onehot_uniques = {}
        self.onehot_cols_by_cate_cols = {}

        self.scalers = {}
        self.conti_cols = []
        self.conti_encoded_cols = []
        self.scale_method = None

        self.cols = []
        self.encoded_cols = []

        self.is_fit = False

    def _fit_cate(self, df):
        for col in df.columns:
            uniques = sorted(list(df[col].unique()))
            self.onehot_uniques[col] = uniques

            onehot_cols = [col + '_onehot_' + str(unique_val) for unique_val in uniques]
            self.onehot_cols_by_cate_cols[col] = onehot_cols
            self.cate_encoded_cols += onehot_cols

    def _fit_conti(self, df, method=None):
        self.scale_method = method
        if self.scale_method is None:
            self.conti_encoded_cols = self.conti_cols
            return df

        for col in df.columns:
            scale_method = self.scale_method_class[method]
            scaler = scale_method()

            np_arr = np.array(df[col]).reshape([-1, 1])
            scaler.fit(np_arr)
            self.scalers[col] = scaler

        self.conti_encoded_cols = [col + f'_{method}_scaled' for col in df.columns]

    def fit(self, df, cate_cols=None, conti_cols=None, scale_method=None):
        if (cate_cols, conti_cols) == (None, None):
            cate_cols = []
            conti_cols = []
            for col in df.columns:
                dtype = df[col].dtype
                if dtype in (np.object, object):
                    cate_cols += [col]
                elif dtype in np_number_types:
                    conti_cols += [col]

        self.cols = cate_cols + conti_cols
        self.cate_cols = cate_cols
        self.conti_cols = conti_cols

        cate_df = df[cate_cols]
        conti_df = df[conti_cols]

        self._fit_cate(cate_df)
        self._fit_conti(conti_df, method=scale_method)
        self.encoded_cols = self.cate_encoded_cols + self.conti_encoded_cols
        self.is_fit = True

    def _encode_cate(self, df: DF):
        n = len(df)

        onehot_df = DF()
        for col in self.cate_cols:
            uniques = self.onehot_uniques[col]
            for unique_val in uniques:
                np_arr = np.zeros(shape=[n])
                np_arr[df[col] == unique_val] = 1
                onehot_df[col + '_onehot_' + str(unique_val)] = np_arr

        return onehot_df

    def _encode_conti(self, df: DF):
        method = self.scale_method
        if method is None:
            return df

        scaled_df = DF()
        for col in df.columns:
            scaler = self.scalers[col]

            np_arr = np.array(df[col]).reshape([-1, 1])
            np_arr = scaler.transform(np_arr)
            scaled_df[col + f'_{method}_scaled'] = np_arr.reshape([-1])

        return scaled_df

    def encode(self, df):
        cate_df = df[self.cate_cols]
        conti_df = df[self.conti_cols]

        cate_df_encoded = self._encode_cate(cate_df)
        conti_df_encoded = self._encode_conti(conti_df)

        concat_df = pd.concat([cate_df_encoded, conti_df_encoded], axis=1)
        concat_df = concat_df[sorted(list(concat_df.columns))]

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
            onehot_cols = self.onehot_cols_by_cate_cols[col]
            decoded_df[col] = self._decode_onehot(df[onehot_cols], uniques)

        return decoded_df

    def decode_conti(self, df: DF):
        if self.scale_method is None:
            return df

        decoded_df = DF()
        for encoded_cols, conti_col in zip(self.conti_encoded_cols, self.conti_cols):
            scaler = self.scalers[conti_col]

            np_arr = np.array(df[encoded_cols]).reshape([-1, 1])
            np_arr = scaler.inverse_transform(np_arr)
            decoded_df[conti_col] = np_arr.reshape([-1])

        return decoded_df

    def decode(self, df: DF):
        cate_encode_df = df[self.cate_encoded_cols]
        cate_df = self.decode_cate(cate_encode_df)

        conti_encode_df = df[self.conti_encoded_cols]
        conti_df = self.decode_conti(conti_encode_df)

        decoded_df = pd.concat([cate_df, conti_df], axis=1)
        decoded_df = decoded_df[self.cols]

        return decoded_df

    def encode_to_np(self, df):
        return self.to_np(self.encode(df))

    def decode_from_np(self, np_arr):
        return self.decode(self.from_np(np_arr, self.encoded_cols))

    def to_np(self, df: DF):
        np_dtypes = [df[key].dtype for key in df.keys()]

        ret = {}
        for key, dtype in zip(df.keys(), np_dtypes):
            np_arr = np.array(df[key].values, dtype=dtype)
            np_arr = np_arr.reshape([len(np_arr), 1])
            ret[key] = np_arr

        return np.concatenate([v for k, v in ret.items()], axis=1)

    def from_np(self, np_arr, np_cols=None):
        if np_cols is None:
            np_cols = self.cols
        print(np_cols)
        print(np_arr.shape)

        df = DF()
        for idx, col in enumerate(np_cols):
            df[col] = np_arr[:, idx]

        return df
