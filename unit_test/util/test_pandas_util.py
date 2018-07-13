from pprint import pprint

import numpy as np

from script.util.PlotTools import PlotTools
from script.util.deco import deco_timeit
from script.util.pandas_util import df_binning
from unit_test.data_handler.Base_df_transformer import DF


def test_df_minmax_normalize():
    x = []
    for i in range(10):
        normal = np.random.normal(i * 2, 1, size=[100])
        x += [normal]
    x = np.concatenate(x)

    df = DF({'x': x})

    pprint(df.head())
    df = df_minmax_normalize(df, 'x')
    pprint(df.head())
    pass


@deco_timeit
def test_df_binning():
    x = []
    for i in range(10):
        normal = np.random.normal(i * 2, 1, size=[100])
        x += [normal]
    x = np.concatenate(x)

    df = DF({'x': x})
    plot = PlotTools(show=True, save=False)
    # plot.dist(df, title='before')

    bin = [-5, 5, 10, 15, 20, 30, 50]

    df = df_binning(df, 'x', bin)

    pprint(df.keys())
    # plot.count(df, 'bucketed_x', title='binning')
    pprint(x)
    pprint(df)


def test_df_to_onehot_embedding():
    x = []
    for i in range(10):
        normal = np.random.normal(i * 2, 1, size=[100])
        x += [normal]
    x = np.concatenate(x)

    df = DF({'x': x})
    plot = PlotTools(show=True, save=False)
    # plot.dist(df, title='before')

    bin = [-5, 5, 10, 15, 20, 30, 50]

    df = df_binning(df, 'x', bin)
    from sklearn.utils import shuffle
    df = shuffle(df)

    pprint(df.head(5))
    df = df_to_onehot_embedding(df[['x_binning']])
    pprint(df.head(5))

    # pprint(df.keys())
    # plot.count(df, 'bucketed_x', title='binning')
    # pprint(x)
    # pprint(df)

    pass