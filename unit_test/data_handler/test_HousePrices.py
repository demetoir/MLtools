from pprint import pprint
import pandas as pd

from script.data_handler.HousePrices import HousePricesHelper
from script.util.deco import deco_timeit
from script.util.misc_util import path_join


@deco_timeit
def test_HousePrices_dataset():
    dataset_path = """C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\data\\HousePrices"""

    merge_df = HousePricesHelper.load_merge_set(dataset_path)

    merge_null_clean = HousePricesHelper.null_cleaning(merge_df)

    merge_type_cast = HousePricesHelper.type_casting(merge_null_clean)

    transformed = HousePricesHelper.transform(merge_type_cast)

    train_df, test_df = HousePricesHelper.train_test_split(transformed)
    train_df.to_csv(path_join(dataset_path, 'transformed_train.csv'), index=False)
    test_df.to_csv(path_join(dataset_path, 'transformed_test.csv'), index=False)
