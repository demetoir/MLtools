import pandas as pd
import numpy as np
import random
import inspect

from script.util.MixIn import LoggerMixIn
from script.util.PlotTools import PlotTools
from script.util.Pool_context import Pool_context

DF = pd.DataFrame
Series = pd.Series


def df_add_col_num(df, zfill_width=None):
    if zfill_width is None:
        zfill_width = 0

    mapping = {}
    for idx, key in enumerate(df.keys()):
        mapping[key] = f'col_{str(idx).zfill(zfill_width)}_{key}'

    return df.rename(mapping, axis='columns')


import_code = """
import pandas as pd
import numpy as np
import random
from script.data_handler.Base_df_null_handler import Base_df_null_handler

DF = pd.DataFrame
Series = pd.Series

"""


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


class Base_df_nullCleaner(LoggerMixIn):
    def __init__(self, df: DF, df_Xs_keys, df_Ys_key, silent=False, verbose=0):
        LoggerMixIn.__init__(self, verbose)
        self.df = df
        self.silent = silent
        self.df_Xs_keys = df_Xs_keys
        self.df_Ys_keys = df_Ys_key
        self.plot = PlotTools()

    def __method_template(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def execute(self, *args, **kwargs) -> DF:
        for key, val in self.__class__.__dict__.items():
            if key in self.df.keys():
                col = self.df[[key]]
                series = self.df[key]
                self.df = val(self, self.df, key, col, series, self.df_Xs_keys, self.df_Ys_keys)

        return self.df

    def gen_info(self):
        with Pool_context() as pool:
            for key, val in list(self.__class__.__dict__.items()):
                if key in self.df.keys():
                    col = self.df[[key]]
                    series = self.df[key]
                    self.print_null_col_info(self.df, key)

                    args = (self.df, key, col, series, self.df_Xs_keys, self.df_Ys_keys)

                    pool.apply_async(self.plot_dist, args=args)
                    pool.apply_async(self.plot_count, args=args)
                    pool.apply_async(self.plot_violin, args=args)
                    pool.apply_async(self.plot_joint2d, args=args)

    @staticmethod
    def drop_col(df: DF, key):
        return df.drop(columns=key)

    @staticmethod
    def fill_major_value_cate(df: DF, key):
        major_value = df[key].astype(str).describe()['top']
        df[key] = df[key].fillna(major_value)
        return df

    @staticmethod
    def fill_random_value_cate(df: DF, key):
        values = df[key].value_counts().keys()
        df[key] = df[key].fillna(lambda x: random.choice(values))
        return df

    @staticmethod
    def fill_rate_value_cate(df: DF, key):
        values, count = zip(*list(df[key].value_counts().items()))
        p = np.array(count) / np.sum(count)
        df[key] = df[key].fillna(lambda x: random.choice(values, p=p))
        return df

    @staticmethod
    def df_null_include(df: DF) -> DF:
        null_column = df.columns[df.isna().any()].tolist()
        return df.loc[:, null_column]

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

    def print_null_col_info(self, df: DF, key):
        if not self.silent:
            col = df[[key]]
            series = df[key]
            print()

            na_count = series.isna().sum()
            total = len(col)
            print(f'column : "{key}", {float(na_count)/float(total):.4f}% {na_count}/{total}(null/total)')
            print(col.describe())

            print('value_counts')
            print(series.value_counts())

            groupby = df[[key, self.df_Ys_keys]].groupby(key).agg(['mean', 'std', 'min', 'max', 'count'])
            print(groupby)

            print()

    @deco_exception_catch
    def plot_dist(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list, path=None):
        np_array = np.array(series[series.notna()])
        self.plot.dist(np_array, title=key, path=f"./matplot/{key}_dist.png")

    @deco_exception_catch
    def plot_count(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list, path=None):
        self.plot.count(df, key, title=key, path=f"./matplot/{key}_count.png")

    @deco_exception_catch
    def plot_violin(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list, path=None):
        self.plot.violin_plot(key, Ys_key, df, title=key, path=f"./matplot/{key}_violin.png")

    @deco_exception_catch
    def plot_joint2d(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list, path=None):
        self.plot.joint_2d(key, Ys_key, df, title=key, path=f'./matplot/{key}_joint2d.png')
