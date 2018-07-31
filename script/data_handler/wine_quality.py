import os
from pprint import pprint
import numpy as np
import pandas as pd
from script.data_handler.Base.BaseDataset import BaseDataset, path_join
from script.data_handler.Base.BaseDatasetPack import BaseDatasetPack
from script.data_handler.Base.Base_df_transformer import Base_df_transformer
from script.util.pandas_util import df_binning, df_minmax_normalize, df_to_onehot_embedding, df_to_np_dict

DF = pd.DataFrame
Series = pd.Series
NpArr = np.array

df_keys = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol', 'color']

df_Xs_keys = [
    'col_0_fixed_acidity', 'col_1_volatile_acidity', 'col_2_citric_acid',
    'col_3_residual_sugar', 'col_4_chlorides', 'col_5_free_sulfur_dioxide',
    'col_6_total_sulfur_dioxide', 'col_7_density', 'col_8_pH',
    'col_9_sulphates', 'col_10_alcohol', 'col_12_color'
]

df_Ys_key = 'col_11_quality'


def cut_hilowend(df):
    col_key = 'col_11_quality'
    df = df[df[col_key] != 3]
    df = df[df[col_key] != 4]
    df = df[df[col_key] != 8]
    df = df[df[col_key] != 9]
    return df


def to_binary_class(df):
    col_key = 'col_11_quality'
    values = [3, 4, 5, 6]
    new_values = 0
    for value in values:
        idxs = df.loc[:, col_key] == value
        df.loc[idxs, col_key] = new_values

    values = [7, 8, 9]
    new_values = 1
    for value in values:
        idxs = df.loc[:, col_key] == value
        df.loc[idxs, col_key] = new_values

    return df


class wine_quality_transformer(Base_df_transformer):
    def col_0_fixed_acidity(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        partial_df = self.mixmax_scale(df, col_key)

        df = self.df_update_col(df, col_key, partial_df)
        return df

    def col_1_volatile_acidity(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        partial_df = self.mixmax_scale(df, col_key)
        df = self.df_update_col(df, col_key, partial_df)
        return df

    def col_2_citric_acid(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        partial_df = self.mixmax_scale(df, col_key)
        df = self.df_update_col(df, col_key, partial_df)
        return df

    def col_3_residual_sugar(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        partial_df = self.mixmax_scale(df, col_key)
        df = self.df_update_col(df, col_key, partial_df)
        return df

    def col_4_chlorides(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        partial_df = self.mixmax_scale(df, col_key)
        df = self.df_update_col(df, col_key, partial_df)
        return df

    def col_5_free_sulfur_dioxide(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list,
                                  Ys_key: list):
        partial_df = self.mixmax_scale(df, col_key)
        df = self.df_update_col(df, col_key, partial_df)
        return df

    def col_6_total_sulfur_dioxide(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list,
                                   Ys_key: list):
        df = df.drop(columns=col_key)
        # partial_df = self.mixmax_normalize(df, col_key)
        # df = self.df_update_col(df, col_key, partial_df)

        return df

    def col_7_density(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        partial_df = self.mixmax_scale(df, col_key)
        df = self.df_update_col(df, col_key, partial_df)
        return df

    def col_8_pH(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        partial_df = self.mixmax_scale(df, col_key)
        df = self.df_update_col(df, col_key, partial_df)
        return df

    def col_9_sulphates(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        partial_df = self.mixmax_scale(df, col_key)
        df = self.df_update_col(df, col_key, partial_df)
        return df

    def col_10_alcohol(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        partial_df = self.mixmax_scale(df, col_key)
        df = self.df_update_col(df, col_key, partial_df)
        return df

    def col_11_quality(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # df = to_binary_class(df)
        # df = cut_hilowend(df)

        # print(df[col_key].value_counts())
        return df

    def col_12_color(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        df = self.drop(df, col_key)
        return df


def load_merge_data(cache=True):
    def df_normalize_column(df, zfill_width=None):
        if zfill_width is None:
            zfill_width = 0

        mapping = {}
        for idx, key in enumerate(df.keys()):
            new_key = key.replace(' ', '_')
            new_key = new_key.replace(' ', '_')
            mapping[key] = f'col_{str(idx).zfill(zfill_width)}_{new_key}'
        pprint(mapping)
        return df.rename(mapping, axis='columns')

    dataset_path = """C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\data\\winequality"""
    merge_path = path_join(dataset_path, 'winequality_merged.csv')
    if not os.path.exists(merge_path) or cache is False:

        wine_red_path = path_join(dataset_path, 'winequality-red.csv')
        wine_white_path = path_join(dataset_path, 'winequality-white.csv')
        wine_red = pd.read_csv(wine_red_path, sep=';', )
        wine_white = pd.read_csv(wine_white_path, sep=';')
        wine_red['color'] = 'red'
        wine_white['color'] = 'white'

        merge_df = pd.concat([wine_red, wine_white], axis=0)
        merge_df = df_normalize_column(merge_df)

        print(merge_df.head())
        print(merge_df.info())

        merge_path = path_join(dataset_path, 'winequality_merged.csv')
        merge_df.to_csv(merge_path, index=False)
    else:
        merge_df = pd.read_csv(merge_path)
    return merge_df


class wine_quality_dataset(BaseDataset):
    def load(self, path, limit=None):
        merge_df = load_merge_data()
        transformer = wine_quality_transformer(merge_df, df_Xs_keys, df_Ys_key)
        merge_df = transformer.transform()

        self.data = df_to_np_dict(merge_df)

    def transform(self):
        df = self.to_DataFrame()
        id_ = pd.DataFrame(df.pop('id_'))
        self.add_data('id_', np.array(id_))

        Ys_df = pd.DataFrame(df.pop(df_Ys_key))
        Ys_df = Ys_df.astype(int)
        Ys_df = df_to_onehot_embedding(Ys_df)
        self.add_data('Ys', np.array(Ys_df))

        Xs_df = df
        self.add_data('Xs', np.array(Xs_df))


class wine_qualityPack(BaseDatasetPack):

    def __init__(self, caching=True, verbose=0, **kwargs):
        super().__init__(caching, verbose, **kwargs)
        self.pack['data'] = wine_quality_dataset(caching=caching, verbose=verbose)
