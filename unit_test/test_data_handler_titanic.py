import pandas as pd
from pprint import pprint
import os


def load_merge_set():
    path = os.getcwd()
    merge_set_path = os.path.join(path, "data", "titanic", "merge_set.csv")
    if not os.path.exists(merge_set_path):
        path = os.getcwd()
        train_path = os.path.join(path, "data", "titanic", "train.csv")
        test_path = os.path.join(path, "data", "titanic", "test.csv")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        merged_df = pd.concat([train_df, test_df])
        merged_df = merged_df.reset_index(drop=True)

        merged_df.to_csv(merge_set_path)
    else:
        merged_df = pd.read_csv(merge_set_path)

    return merged_df


def trans_cabin(df):
    df = pd.DataFrame(df['Cabin'])

    cabin_head = df.query('not Cabin.isna()')
    cabin_head = cabin_head['Cabin'].astype(str)
    cabin_head = cabin_head.str.slice(0, 1)

    na = df.query('Cabin.isna()')

    df.loc[na.index, 'Cabin'] = 'None'
    df.loc[cabin_head.index, 'Cabin'] = cabin_head
    return df


def test_trans_cabin():
    merge_set_df = load_merge_set()
    cabin = trans_cabin(merge_set_df)

    pprint(cabin.head(5))
    pprint(cabin.info())
