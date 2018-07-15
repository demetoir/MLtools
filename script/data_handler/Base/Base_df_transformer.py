import pandas as pd
import inspect
from script.data_handler.Base.df_plotterMixIn import df_plotterMixIn
from script.util.MixIn import LoggerMixIn
import numpy as np

from script.util.PlotTools import PlotTools
from script.util.pandas_util import df_binning, df_minmax_normalize, df_to_onehot_embedding

DF = pd.DataFrame
Series = pd.Series
NpArr = np.array


class transform_methodMixIn:

    @staticmethod
    def mixmax_normalize(df: DF, col: str) -> DF:
        return df_minmax_normalize(df, col)

    @staticmethod
    def binning(df: DF, col: str, bin_seq: list, column_tail='_binning') -> DF:
        return df_binning(df, col, bin_seq, column_tail)

    @staticmethod
    def to_onehot(df: DF, col: list) -> DF:
        return df_to_onehot_embedding(df[col])


class Base_df_transformer(LoggerMixIn, df_plotterMixIn, transform_methodMixIn):
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

    def boilerplate_maker(self, path=None, encoding='UTF8'):
        base_class_name = super().__class__.__name__

        import_code = f"""
        import pandas as pd
        import numpy as np
        import random
        from script.data_handler.{base_class_name} import {base_class_name} 

        DF = pd.DataFrame
        Series = pd.Series

        """
        code = [import_code]

        base_class_name = self.__class__.__name__
        class_name = f"boiler_plate_{base_class_name}"
        class_template = f"""class {class_name}({base_class_name}):"""
        code += [class_template.format(class_name=class_name)]

        method_template = inspect.getsource(self.__method_template)
        method_template = method_template.replace('__method_template', '{col_name}')
        for key in self.df.keys():
            method_code = method_template.format(col_name=key)
            code += [method_code]

        code = "\n".join(code)
        if path is not None:
            with open(path, mode='w', encoding=encoding) as f:
                f.write(code)

        return code

    def plot_all(self):
        self._df_cols_plot(self.df, self.df_Xs_keys, self.df_Ys_key)

    def corr_heatmap(self):
        plot = PlotTools(save=False, show=True)

        from scipy.stats import pearsonr
        keys = self.df.keys()

        # corr = self.df.corr()
        # print(corr.info())
        # plot.heatmap(corr)

    def _execute_method(self, caller_name) -> DF:
        for key, func in self.__class__.__dict__.items():
            if key in self.df.keys():
                col = self.df[[key]]
                series = self.df[key]

                self.df = func(self, self.df, key, col, series, self.df_Xs_keys, self.df_Ys_key)

        return self.df

    def transform(self):
        return self._execute_method('transform')
