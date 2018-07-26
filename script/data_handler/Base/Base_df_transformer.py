import inspect
import pandas as pd
import numpy as np
from script.data_handler.Base.df_plotterMixIn import df_plotterMixIn
from script.util.MixIn import LoggerMixIn
from script.util.PlotTools import PlotTools
from script.util.pandas_util import df_binning, df_minmax_normalize, df_to_onehot_embedding
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

DF = pd.DataFrame
Series = pd.Series
NpArr = np.array


class df_PreprocessingMixIn:
    @staticmethod
    def MinMaxScaler(df, x_col, y_col):
        pass

    @staticmethod
    def MaxAbsScaler():
        pass

    @staticmethod
    def RobustScaler():
        pass

    @staticmethod
    def StandardScaler():
        pass


class transform_methodMixIn:

    @staticmethod
    def mixmax_scale(df: DF, col: str) -> DF:
        return df_minmax_normalize(df, col)

    @staticmethod
    def binning(df: DF, col: str, bin_seq: list, column_tail='_binning', with_intensity=False) -> DF:
        return df_binning(df, col, bin_seq, column_tail, with_intensity=with_intensity)

    @staticmethod
    def to_onehot(df: DF, col: list) -> DF:
        return df_to_onehot_embedding(df[col])

    @staticmethod
    def df_update_col(df, old_column, new_df):
        df = df.reset_index(drop=True)
        new_df = new_df.reset_index(drop=True)

        df = df.drop(columns=old_column)
        df = pd.concat([df, new_df], axis=1)
        return df

    @staticmethod
    def df_concat(df, new_df):
        df = df.reset_index(drop=True)
        new_df = new_df.reset_index(drop=True)

        df = pd.concat([df, new_df], axis=1)
        return df

    @staticmethod
    def df_group_values(values, new_values, df, col_key):
        for value in values:
            idxs = df.loc[:, col_key] == value
            df.loc[idxs, col_key] = new_values

        return df

    @staticmethod
    def drop(df: DF, col: str):
        return df.drop(columns=col)

    @staticmethod
    def Imputer(df, x_col, y_col=None, missing_values="NaN", strategy="mean"):
        imputer = Imputer(missing_values=missing_values, strategy=strategy)
        imputer.fit(x_col, y_col)
        df[x_col] = imputer.transform(x_col)
        return df

    @staticmethod
    def _Encoder_common(enc, df, col, with_mapper=False):

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

    @staticmethod
    def LabelEncoder(df, col, with_mapper=False):
        return transform_methodMixIn._Encoder_common(
            LabelEncoder(),
            df,
            col,
            with_mapper=with_mapper
        )

    @staticmethod
    def OnehotEncoder(df, col, with_mapper=False):
        return transform_methodMixIn._Encoder_common(
            OneHotEncoder(),
            df,
            col,
            with_mapper=with_mapper
        )


class Base_df_transformer(LoggerMixIn, df_plotterMixIn, transform_methodMixIn):
    import_code = f"""
    import pandas as pd
    import numpy as np
    import random
    from script.data_handler.Base_df_transformer import Base_df_transformer

    DF = pd.DataFrame
    Series = pd.Series
"""
    class_code = """class boiler_plate(Base_df_transformer):"""

    def __init__(self, df: DF, df_Xs_keys, df_Ys_key, silent=False, verbose=0):
        LoggerMixIn.__init__(self, verbose)
        df_plotterMixIn.__init__(self)
        transform_methodMixIn.__init__(self)

        self.df = df
        self.df_Xs_keys = df_Xs_keys
        self.df_Ys_key = df_Ys_key
        self.silent = silent

    def __method_template(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    @property
    def method_template(self):
        func = self.__method_template
        method_template = inspect.getsource(func)
        method_template = method_template.replace(func.__name__, '{col_name}')
        return method_template

    def boilerplate_maker(self, path=None, encoding='UTF8'):
        code = [self.import_code]
        code += [self.class_code]

        for key in self.df.keys():
            code += [self.method_template.format(col_name=key)]

        code = "\n".join(code)

        if path is not None:
            with open(path, mode='w', encoding=encoding) as f:
                f.write(code)

        return code

    def corr_heatmap(self):
        plot = PlotTools(save=False, show=True)
        corr = self.df.corr()
        plot.heatmap(corr)

    def transform(self):
        for key, func in self.__class__.__dict__.items():
            if key in self.df.keys():
                col = self.df[[key]]
                series = self.df[key]

                self.df = func(self, self.df, key, col, series, self.df_Xs_keys, self.df_Ys_key)

        for key, func in self.__class__.__dict__.items():
            if callable(func) and 'col_new_' in key:
                ret = func(self, self.df, self.df_Xs_keys, self.df_Ys_key)
                if ret is None:
                    raise ValueError
                self.df = ret

        self.df = self.df.sort_index(axis=1)

        # TODO rename_col_num
        # column = self.df.columns
        # for col in column:
        #     if 'col_new_' in col:
        #
        # print(column)

        return self.df
