import pandas as pd
import numpy as np
import inspect

from script.data_handler.Base.df_plotterMixIn import df_plotterMixIn
from script.util.MixIn import LoggerMixIn
from script.util.PlotTools import PlotTools

DF = pd.DataFrame
Series = pd.Series


class null_clean_methodMixIn:
    @staticmethod
    def drop_col(df: DF, key):
        return df.drop(key, axis=1)

    @staticmethod
    def fill_major_value_cate(df: DF, key) -> DF:
        major_value = df[key].astype(str).describe()['top']
        df[key] = df[key].fillna(major_value)
        return df

    @staticmethod
    def fill_random_value_cate(df: DF, key) -> DF:
        values = df[key].value_counts().keys()
        df[key] = df[key].transform(lambda x: x.fillna(np.random.choice(values)))
        # df[key] = df[key].fillna()
        return df

    @staticmethod
    def fill_rate_value_cate(df: DF, key) -> DF:
        values, count = zip(*list(df[key].value_counts().items()))
        p = np.array(count) / np.sum(count)
        df[key] = df[key].transform(lambda x: x.fillna(np.random.choice(values, p=p)))
        return df


class Base_dfCleaner(LoggerMixIn, null_clean_methodMixIn, df_plotterMixIn):
    import_code = """
    import pandas as pd
    import numpy as np
    import random
    from script.data_handler.Base_dfCleaner import Base_dfCleaner 

    DF = pd.DataFrame
    Series = pd.Series
    
"""

    class_template = """
class dfCleaner(Base_dfCleaner):
"""

    def __init__(self, df: DF, df_Xs_keys, df_Ys_key, silent=False, verbose=0):
        LoggerMixIn.__init__(self, verbose)
        null_clean_methodMixIn.__init__(self)
        df_plotterMixIn.__init__(self)

        self.df = df
        self.silent = silent
        self.df_Xs_keys = df_Xs_keys
        self.df_Ys_key = df_Ys_key
        self.plot = PlotTools()

    def __method_template(self, df: DF, col_key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    @property
    def method_template(self):
        method_template = inspect.getsource(self.__method_template)
        method_template = method_template.replace('__method_template', '{col_name}')
        return method_template

    def boilerplate_maker(self, path=None, encoding='UTF8'):
        code = [self.import_code]
        code += [self.class_template]

        df_only_null = self._df_null_include(self.df)
        for key in df_only_null.keys():
            code += [self.method_template.format(col_name=key)]

        code = "\n".join(code)
        if path is not None:
            with open(path, mode='w', encoding=encoding) as f:
                f.write(code)

        return code

    def clean(self) -> DF:
        for key, val in self.__class__.__dict__.items():
            if key in self.df.keys():
                col = self.df[[key]]
                series = self.df[key]

                self.df = val(self, self.df, key, col, series, self.df_Xs_keys, self.df_Ys_key)

        return self.df

    def null_cols_info(self) -> str:
        ret = []
        for key, val in list(self.__class__.__dict__.items()):
            if key in self.df.keys():
                info = self._str_null_col_info(self.df, key)
                ret += [info]

        return "\n\n".join(ret)

    def null_cols_plot(self):
        df_only_null = self._df_null_include(self.df)
        self._df_cols_plot(df_only_null, list(df_only_null.keys()), self.df_Ys_key)

    @staticmethod
    def _df_null_include(df: DF) -> DF:
        null_column = df.columns[df.isna().any()].tolist()
        return df.loc[:, null_column]

    def _str_null_col_info(self, df: DF, key) -> str:
        ret = []
        col = df[[key]]
        series = df[key]

        na_count = series.isna().sum()
        total = len(col)
        ret += [f'column : "{key}", null ratio:{float(na_count)/float(total):.4f}%,  {na_count}/{total}(null/total)']
        ret += [col.describe()]
        ret += ['value_counts']
        ret += [series.value_counts()]
        groupby = df[[key, self.df_Ys_key]].groupby(key).agg(['mean', 'std', 'min', 'max', 'count'])
        ret += [groupby]

        return "\n".join(map(str, ret))
