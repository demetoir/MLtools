from pprint import pprint
import numpy as np
import pandas as pd
from script.util.deco import deco_timeit
from script.util.numpy_utils import np_equal_bins

DF = pd.DataFrame

@deco_timeit
def test_np_equal_bins():
    x = [np.random.normal(i * 2, 1, size=[1000000]) for i in range(5)]
    x = np.concatenate(x)

    bins = np_equal_bins(x, 100)
    pprint(bins)

    # df = DF({'x': x})
    # binning_df = df_binning(df, 'x', list(bins.tolist()))
    #
    # plot = PlotTools(show=True, save=False)
    # plot.count(binning_df, 'x_binning')
    # pprint(binning_df['x_binning'].unique())
