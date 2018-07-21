from script.util.deco import deco_timeit
import numpy as np
import pandas as pd


@deco_timeit
def assign(n):
    a = [i for i in range(100)]
    for _ in range(n):
        idx = [i for i in range(50)]
        a = idx


@deco_timeit
def direct(n):
    a = [i for i in range(100)]
    for _ in range(n):
        a = [i for i in range(50)]


def test_timing_assign_vs_direct():
    n = 1000000
    assign(n)
    direct(n)


def test_timeit_np_arr_to_pd_df():
    size = 32 * 32
    np_arr = np.ones([32, 32])

    @deco_timeit
    def to_df(np_arr):
        df = pd.Series(np_arr)
        df = df.to_frame('np_arr')

        return df

    @deco_timeit
    def to_series(np_arr):
        df = pd.Series(np_arr)
        return df

    df = to_df(np_arr)
    series = to_series(np_arr)
    print(series)
