from script.util.PlotTools import PlotTools
from script.util.JobPool import JobPool
import pandas as pd

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
    def _plot_dist(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_keys: list, Ys_key: list, path=None):
        title = f'{col_key}_plot_dist'
        self.plot.dist(df, col_key, title=title, path=f"./matplot/{title}.png")

    @deco_exception_catch
    def _plot_count(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_keys: list, Ys_key: list, path=None):
        title = f'{col_key}_plot_count_bar'
        self.plot.count(df, col_key, title=title, path=f"./matplot/{title}.png")

    @deco_exception_catch
    def _plot_violin(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_keys: list, Ys_key: list,
                     path=None):
        title = f'{col_key}_plot_violin'
        self.plot.violin_plot(df, col_key, Ys_key, path=f"./matplot/{title}_1.png", title=title)
        self.plot.violin_plot(df, Ys_key, col_key, path=f"./matplot/{title}_2.png", title=title)

    @deco_exception_catch
    def _plot_joint2d(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_keys: list, Ys_key: list,
                      path=None):
        title = f'{col_key}_plot_joint2d'
        self.plot.joint_2d(df, col_key, Ys_key, path=f"./matplot/{title}.png", title=title)

    @deco_exception_catch
    def _plot_dist_groupby(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_keys: list, Ys_key: list,
                           path=None):
        title = f'{col_key}_plot_dist_groupby'
        self.plot.dist_groupby(df, Ys_key, col_key, df, title=title, path=f"./matplot/{title}.png")
        self.plot.dist_groupby(df, col_key, Ys_key, df, title=title, path=f"./matplot/{title}.png")

    def _df_cols_plot(self, df, df_Xs_keys, df_Ys_key):
        with JobPool() as pool:
            for key in df_Xs_keys:
                col = df[[key]]
                series = df[key]
                args = (df, key, col, series, df_Xs_keys, df_Ys_key)

                pool.apply_async(self._plot_dist, args=args)
                pool.apply_async(self._plot_count, args=args)
                pool.apply_async(self._plot_violin, args=args)
                pool.apply_async(self._plot_joint2d, args=args)
