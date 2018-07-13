from pprint import pprint
import pandas as pd
from script.data_handler.HousePrices import load_merge_set, null_cleaning, transform_df, train_test_split
from script.util.misc_util import path_join


def test_train_test_split():
    dataset_path = """C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\data\\HousePrices"""

    merge_df = load_merge_set(dataset_path)

    merge_null_clean = null_cleaning(merge_df)
    transformed = transform_df(merge_null_clean)
    train_df, test_df = train_test_split(transformed)
    test_df = test_df.reset_index()

    train_path = path_join(dataset_path, 'train.csv')
    train_origin = pd.read_csv(train_path)

    test_path = path_join(dataset_path, 'test.csv')
    test_origin = pd.read_csv(test_path)
    test_origin = test_origin.reset_index()

    pprint(train_origin[['1stFlrSF']].head(5))
    pprint(train_df[['col_00_1stFlrSF']].head(5))

    pprint(test_origin[['1stFlrSF']].head(5))
    pprint(test_df[['col_00_1stFlrSF']].head(5))

    assert train_origin['1stFlrSF'].equals(train_df['col_00_1stFlrSF'])
    assert test_origin['1stFlrSF'].equals(test_df['col_00_1stFlrSF'])
