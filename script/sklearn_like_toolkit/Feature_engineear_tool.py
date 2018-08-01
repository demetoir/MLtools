import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
from tqdm import trange

from script.data_handler.Base.Base_df_transformer import DF
from script.util.pandas_util import df_minmax_normalize, df_binning, df_to_onehot_embedding


class scale_method:
    @staticmethod
    def MinMaxScaler(df, x_col):
        scale = preprocessing.MinMaxScaler()
        scale.fit(df[x_col])
        return scale.transform(df[x_col])

    @staticmethod
    def MaxAbsScaler(df, x_col):
        scale = preprocessing.MaxAbsScaler()
        scale.fit(df[x_col])
        return scale.transform(df[x_col])

    @staticmethod
    def RobustScaler(df, x_col):
        scale = preprocessing.RobustScaler()
        scale.fit(df[x_col])
        return scale.transform(df[x_col])

    @staticmethod
    def StandardScaler(df, x_col):
        scale = preprocessing.StandardScaler()
        scale.fit(df[x_col])
        return scale.transform(df[x_col])


class Impute_method:
    def Imputer(self, df, x_col, missing_values="NaN", strategy="mean"):
        imputer = Imputer(missing_values=missing_values, strategy=strategy)

        return DF({
            x_col + 'impute_' + strategy: imputer.fit(df[x_col])

        })

    def _Imputer(self, df, x_col, y_col=None, missing_values="NaN", strategy="mean"):
        imputer = Imputer(missing_values=missing_values, strategy=strategy)
        imputer.fit(x_col, y_col)
        df[x_col] = imputer.transform(x_col)

        """        - If "mean", then replace missing values using the mean along
          the axis.
        - If "median", then replace missing values using the median along
          the axis.
        - If "most_frequent", then replace missing using the most frequent
          value along the axis."""
        return df

    def Imuter_mean(self, df, x_col, y_col=None, missing_values='nan'):
        return self._Imputer(df, x_col, y_col, missing_values, strategy='mean')

    def Imuter_median(self, df, x_col, y_col=None, missing_values='nan'):
        return self._Imputer(df, x_col, y_col, missing_values)

    def Imuter_most_frequent(self, df, x_col, y_col=None, missing_value='nan'):
        return self._Imputer(df, x_col, y_col, missing_value)


class label_encode:
    def _Encoder_common(self, enc, df, col, with_mapper=False):

        col_encode = col + "_encoded"
        enc.fit(df[col])

        new_df = DF({
            col_encode: enc.transform(df[col])
        })

        if with_mapper:
            unique = df[col].unique()
            mapped = enc.transform(unique)
            encoder = {zip(unique, mapped)}
            decoder = {zip(mapped, unique)}

            return new_df, encoder, decoder
        else:

            return new_df

    def LabelEncoder(self, df, col, with_mapper=False):
        return self._Encoder_common(
            LabelEncoder(),
            df,
            col,
            with_mapper=with_mapper
        )

    def OnehotEncoder(self, df, col, with_mapper=False):
        return self._Encoder_common(
            OneHotEncoder(),
            df,
            col,
            with_mapper=with_mapper
        )


class Feature_engineer_tool(label_encode, Impute_method, scale_method):
    def corr_maximize_bins(self, df, x_col, y_col, n_iter, size):
        best = 0
        best_bins = None
        for _ in trange(n_iter):
            seed = np.arange(min(df[x_col]), max(df[x_col]), 0.1)
            rand_bins = np.random.choice(seed, size=size)

            bins = [min(df[x_col]) - 1] + list(sorted(rand_bins)) + [max(df[x_col]) + 1]
            col_binning = x_col + '_binning'
            binning_df = self.binning(df, x_col, bins)

            col_encode = col_binning + '_encoded'
            encoding_df = self.LabelEncoder(binning_df, col_binning)
            part = self.concat_df(df[[y_col]], encoding_df)

            corr = DF(part.corr())
            new_val = float(corr.loc[y_col, col_encode])
            if best < np.abs(new_val):
                best = np.abs(new_val)
                best_bins = bins

        return best_bins, best

    def mixmax_scale(self, df: DF, col: str) -> DF:
        return df_minmax_normalize(df, col)

    def binning(self, df: DF, col: str, bin_seq: list, column_tail='_binning', with_intensity=False) -> DF:
        return df_binning(df, col, bin_seq, column_tail, with_intensity=with_intensity)

    def to_onehot(self, df: DF, col: list) -> DF:
        return df_to_onehot_embedding(df[col])

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