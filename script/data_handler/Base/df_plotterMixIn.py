from script.util.PlotTools import PlotTools
import numpy as np
import pandas as pd

from script.util.Pool_context import Pool_context

DF = pd.DataFrame
Series = pd.Series


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

    def plot_all(self, df, df_Xs_keys, df_Ys_key):
        self._df_cols_plot(df, df_Xs_keys, df_Ys_key)

    @deco_exception_catch
    def _plot_dist(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list, path=None):
        np_array = np.array(series[series.isna()])
        title = f'{key}_plot_dist'
        self.plot.dist(df, key, title=title, path=f"./matplot/{title}.png")

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
        # raise TypeError()

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
