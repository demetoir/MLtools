import numpy as np
import pandas as pd
from script.util.numpy_utils import np_minmax_normalize

DF = pd.DataFrame
NpArr = np.array


def df_bucketize(df, key, bucket_range, column='bucket', na='None', null='None'):
    new_df = pd.DataFrame({column: df[key]})

    for i in range(len(bucket_range) - 1):
        a, b = bucket_range[i: i + 2]
        name = f'{a}~{b}'

        query = f'{a} <= {key} < {b}'

        idx = list(df.query(query).index.values)
        new_df.loc[idx] = name

    idx = list(df.query(f'{key}.isna()').index.values)
    new_df.loc[idx] = na

    idx = list(df.query(f'{key}.isnull()').index.values)
    new_df.loc[idx] = null

    return new_df


def df_to_np_dict(df, dtype=None):
    ret = {}
    for key in df.keys():
        np_arr = np.array(df[key].values, dtype=dtype)
        np_arr = np_arr.reshape([len(np_arr), 1])
        ret[key] = np_arr
    return ret


def df_to_onehot_embedding(df: DF) -> DF:
    ret = pd.DataFrame({'_idx': np.zeros(len(df))})

    for df_key in df.keys():
        np_arr = np.array(df[df_key])
        for unique_key in sorted(df[df_key].unique()):
            ret[f'{df_key}_{unique_key}'] = np.array(np.where(np_arr == unique_key, 1, 0).reshape([-1, 1]))

    ret = ret.drop(columns=['_idx'])
    return ret


def df_to_np_onehot_embedding(df):
    ret = {}
    for df_key in df.keys():
        np_arr = np.array(df[df_key])
        for unique_key in df[df_key].unique():
            ret[f'{df_key}_{unique_key}'] = np.where(np_arr == unique_key, 1, 0).reshape([-1, 1])

    ret = np.concatenate([v for k, v in ret.items()], axis=1)
    return ret


def df_binning(df: DF, key: str, bin_seq: list, column_tail='_binning') -> DF:
    col_bucketed = key + column_tail
    col_intensity = key + '_intensity'
    new_df = pd.DataFrame({
        col_bucketed: df[key],
        col_intensity: np.zeros(len(df[key]))
    })

    for i in range(len(bin_seq) - 1):
        a, b = bin_seq[i: i + 2]
        name = f'{a}~{b}'

        query = f'{a} <= {key} < {b}'
        idx = list(df.query(query).index.values)
        new_df.loc[idx, col_bucketed] = name
        new_df.loc[idx, col_intensity] = (df.loc[idx, key] - a) / (b - a)

    return new_df


def df_minmax_normalize(df: DF, key: str, min=None, max=None, col_tail='_normailized') -> DF:
    np_x = NpArr(df[key])
    np_x_normalized = np_minmax_normalize(np_x, min, max)
    new_df = DF({key + col_tail: np_x_normalized})
    return new_df
