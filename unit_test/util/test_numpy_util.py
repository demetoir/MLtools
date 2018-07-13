from pprint import pprint
import numpy as np
import pandas as pd
from script.util.deco import deco_timeit
from script.util.numpy_utils import np_frequency_equal_bins, np_width_equal_bins
from script.util.pandas_util import df_binning

DF = pd.DataFrame


@deco_timeit
def test_np_frequency_equal_bins():
    x = [np.random.normal(i * 2, 1, size=[1000000]) for i in range(5)]
    x = np.concatenate(x)

    bins = np_frequency_equal_bins(x, 100)
    pprint(bins)

    # df = DF({'x': x})
    # binning_df = df_binning(df, 'x', list(bins.tolist()))
    #
    # plot = PlotTools(show=True, save=False)
    # plot.count(binning_df, 'x_binning')
    # pprint(binning_df['x_binning'].unique())


@deco_timeit
def test_np_width_equal_bins():
    x = [np.random.normal(i * 2, 1, size=[100]) for i in range(5)]
    x = np.concatenate(x)
    print(min(x), max(x))

    bins = np_width_equal_bins(x, 2)
    pprint(bins)
    # plot = PlotTools(save=False, show=True)

    df = DF({'x': x})
    binned_df = df_binning(df, 'x', list(bins.tolist()))

    # plot.count(binned_df, 'x_binning')
