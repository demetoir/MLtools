import numpy as np
import pandas as pd

from script.util.asyncPlotTools import asyncPlotTools


def test_asyncPlotTools():
    x = np.array([1, 2, 3, 4])
    x = x.reshape([2, 2])

    df = pd.DataFrame({
        'X': np.arange(0, 10000, 1)
    })
    plot = asyncPlotTools(n_job=2, save=True, show=False)

    plot.dist(df, 'X')
    plot.dist(df, 'X')
    plot.dist(df, 'X')
    plot.dist(df, 'X')
    plot.dist(df, 'X')

    plot.join()
