from script.util.MixIn import LoggerMixIn
from script.util.PlotTools import PlotTools
from script.util.JobPool import JobPool
import pandas as pd

from script.util.misc_util import log_error_trace

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
            log_error_trace(self.log.warn, e)

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


class DF_PlotTools(LoggerMixIn):
    def __init__(self, df, y_key, n_job=7, path=None):
        super().__init__()
        self.plot = PlotTools()
        self.n_job = n_job
        self.df = df
        self.y_key = y_key

        self.path = path
        if self.path is None:
            self.path = "./plot_outs"

    def plot_all(self, ):
        with JobPool(self.n_job) as pool:
            keys = self.df.keys()
            df = self.df

            # plot dist
            for key in keys:
                args = [df, key]
                pool.apply_async(self.plot_dist, args)

            # plot count
            for key in keys:
                args = [df, key]
                pool.apply_async(self.plot_countbar, args)

            # plot violin
            for key in keys:
                args = [df, key, self.y_key]
                pool.apply_async(self.plot_violin, args)

                args = [df, self.y_key, key]
                pool.apply_async(self.plot_violin, args)

            # plot_joint2d
            for key in keys:
                args = [df, key, self.y_key]
                pool.apply_async(self.plot_joint2d, args)

                args = [df, self.y_key, key]
                pool.apply_async(self.plot_joint2d, args)

            # plot countbar_groupby
            for a_key in keys:
                args = [df, a_key, self.y_key]
                pool.apply_async(self.plot_count_groupby, args)

    @deco_exception_catch
    def plot_dist(self, df: DF, col_key: str, title=None):
        if title is None:
            title = f'{col_key}_plot_dist'

        self.plot.dist(df, col_key, title=title, path=f"./matplot/{title}.png")

    @deco_exception_catch
    def plot_countbar(self, df: DF, col_key: str, title=None):
        if title is None:
            title = f'{col_key}_plot_count_bar'

        self.plot.count(df, col_key, title=title, path=f"./matplot/{title}.png")

    @deco_exception_catch
    def plot_violin(self, df: DF, a_col, b_col, title=None):
        if title is None:
            title = f'{a_col}_{b_col}_plot_violin'

        self.plot.violin_plot(df, a_col, b_col, path=f"./matplot/{title}.png", title=title)

    @deco_exception_catch
    def plot_joint2d(self, df: DF, a_col, b_col, title=None):
        if title is None:
            title = f'{a_col}_{b_col}_plot_joint2d'

        self.plot.joint_2d(df, a_col, b_col, path=f"./matplot/{title}.png", title=title)

    @deco_exception_catch
    def plot_dist_groupby(self, df: DF, a_col, groupby_cols, title=None):
        if title is None:
            title = f'{a_col}_groupby_{groupby_cols}_plot_dist_groupby'

        self.plot.dist_groupby(df, a_col, groupby_cols, title=title, path=f"./matplot/{title}.png")

    @deco_exception_catch
    def plot_count_groupby(self, df: DF, a_col, groupby_cols, title=None):
        if title is None:
            title = f'{a_col}_groupby_{groupby_cols}_count'

        self.plot.count(df, a_col, groupby_cols, title=title, path=f"./matplot/{title}.png")

    @deco_exception_catch
    def plot_violin_groupby(self, df: DF, a_col, b_col, groupby_cols, title=None):
        if title is None:
            title = f'{a_col}_{b_col}_groupby_{groupby_cols}_violin'

        self.plot.violin_plot(df, a_col, b_col, groupby_cols, title=title, path=f"./matplot/{title}.png")
