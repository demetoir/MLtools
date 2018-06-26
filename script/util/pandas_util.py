import numpy as np
import pandas as pd


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


def df_to_onehot_embedding(df):
    ret = pd.DataFrame({'_idx': [i for i in range(len(df))]})
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
