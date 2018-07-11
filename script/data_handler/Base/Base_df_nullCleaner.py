import pandas as pd
import numpy as np
import random
import inspect

from script.util.MixIn import LoggerMixIn
from script.util.PlotTools import PlotTools
from script.util.Pool_context import Pool_context

DF = pd.DataFrame
Series = pd.Series

import_code = """
import pandas as pd
import numpy as np
import random
from script.data_handler.Base_df_null_handler import Base_df_null_handler

DF = pd.DataFrame
Series = pd.Series

"""


class null_clean_methodMixIn:
    @staticmethod
    def drop_col(df: DF, key) -> DF:
        return df.drop(columns=key)

    @staticmethod
    def fill_major_value_cate(df: DF, key) -> DF:
        major_value = df[key].astype(str).describe()['top']
        df[key] = df[key].fillna(major_value)
        return df

    @staticmethod
    def fill_random_value_cate(df: DF, key) -> DF:
        values = df[key].value_counts().keys()
        df[key] = df[key].fillna(lambda x: random.choice(values))
        return df

    @staticmethod
    def fill_rate_value_cate(df: DF, key) -> DF:
        values, count = zip(*list(df[key].value_counts().items()))
        p = np.array(count) / np.sum(count)
        df[key] = df[key].fillna(lambda x: random.choice(values, p=p))
        return df


def deco_exception_catch(func):
    def wrapper(*args, **kwargs):
        self = args[0]
        key = args[2]

        try:
            return func(*args, **kwargs)
        except BaseException as e:
            self.log.warn(f'\nfail {func.__name__}, {key}\n')
            print(e)

    return wrapper


class df_plotterMixIn:

    def __init__(self):
        self.plot = PlotTools()

    @deco_exception_catch
    def _plot_dist(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list, path=None):
        # np_array = np.array(series)

        title = f'{key}_plot_dist'
        self.plot.dist(np.array(series), title=title, path=f"./matplot/{title}.png")

    @deco_exception_catch
    def _plot_count(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list, path=None):
        title = f'{key}_plot_count_bar'
        self.plot.count(df, key, title=title, path=f"./matplot/{title}.png")

    @deco_exception_catch
    def _plot_violin(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list, path=None):
        # df[key] = df[df[key].notna().to_list()]
        # df[Ys_key] = df[df[key].notna()]
        title = f'{key}_plot_violin'
        self.plot.violin_plot(key, Ys_key, df, title=title, path=f"./matplot/{title}.png")

    @deco_exception_catch
    def _plot_joint2d(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list, path=None):
        if df[key].dtype is not float:
            raise TypeError()

        title = f'{key}_plot_joint2d'
        self.plot.joint_2d(key, Ys_key, df, title=title, path=f"./matplot/{title}.png")

    def _df_cols_plot(self, df, df_Xs_keys, df_Ys_key):
        with Pool_context() as pool:
            for key in list(df.keys()):
                col = df[[key]]
                series = df[key]
                args = (df, key, col, series, df_Xs_keys, df_Ys_key)

                pool.apply_async(self._plot_dist, args=args)
                pool.apply_async(self._plot_count, args=args)
                pool.apply_async(self._plot_violin, args=args)
                pool.apply_async(self._plot_joint2d, args=args)


class Base_df_nullCleaner(LoggerMixIn, null_clean_methodMixIn, df_plotterMixIn):
    def __init__(self, df: DF, df_Xs_keys, df_Ys_key, silent=False, verbose=0):
        LoggerMixIn.__init__(self, verbose)
        null_clean_methodMixIn.__init__(self)
        df_plotterMixIn.__init__(self)

        self.df = df
        self.silent = silent
        self.df_Xs_keys = df_Xs_keys
        self.df_Ys_key = df_Ys_key
        self.plot = PlotTools()

    def __method_template(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def clean_null(self) -> DF:
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

    def df_null_include(self, df: DF) -> DF:
        null_column = df.columns[df.isna().any()].tolist()
        return df.loc[:, null_column]

    def null_cols_plot(self):
        df_only_null = self.df_null_include(self.df)
        self._df_cols_plot(df_only_null, list(df_only_null.keys()), self.df_Ys_key)

    def boilerplate_maker(self, path=None, encoding='UTF8'):

        class_name = "boilder_plate_Base_df_Null_handler"
        class_template = """class {class_name}(Base_df_null_handler):"""
        mothod_template = inspect.getsource(self.__method_template)
        mothod_template = mothod_template.replace('__method_template', '{col_name}')

        code = [
            import_code,
            class_template.format(class_name=class_name)
        ]

        df_only_null = self.df_null_include(self.df)
        for key in df_only_null.keys():
            method_code = mothod_template.format(col_name=key)
            code += [method_code]

        code = "\n".join(code)
        if path is not None:
            with open(path, mode='w', encoding=encoding) as f:
                f.write(code)

        return code

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




