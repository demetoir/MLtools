import os
import pandas as pd
from script.data_handler.Base.BaseDataset import BaseDataset
from script.data_handler.Base.BaseDatasetPack import BaseDatasetPack
from script.data_handler.Base.Base_dfCleaner import Base_dfCleaner
from script.data_handler.Base.Base_df_transformer import Base_df_transformer
from script.data_handler.Base.base_df_typecasting import base_df_typecasting
from script.util.misc_util import path_join
from script.util.numpy_utils import *
from script.util.pandas_util import df_to_np_dict, df_to_onehot_embedding, df_to_np_onehot_embedding

DF = pd.DataFrame
Series = pd.Series

FOLDER_NAME = "tatanic"
PASSENGERID = 'PassengerId'
SURVIVED = 'Survived'
PCLASS = 'Pclass'
NAME = 'Name'
SEX = 'Sex'
AGE = 'Age'
SIBSP = 'SibSp'
PARCH = 'Parch'
TICKET = 'Ticket'
FARE = 'Fare'
CABIN = 'Cabin'
EMBARKED = 'Embarked'
Xs = 'Xs'
Ys = 'Ys'

df_Xs_keys = [
    'col_00_Age', 'col_01_Cabin', 'col_02_Embarked', 'col_03_Fare',
    'col_04_Name', 'col_05_Parch', 'col_06_PassengerId', 'col_07_Pclass',
    'col_08_Sex', 'col_09_SibSp', 'col_11_Ticket'
]

df_Ys_key = 'col_10_Survived'

# TODO remake

def np_str_labels_to_index(np_arr, labels):
    np_arr = np.asarray(np_arr)
    new_arr = np.zeros_like(np_arr)
    for idx, label in enumerate(labels):
        new_arr = np.where(np_arr == label, idx, new_arr)

    return new_arr.astype(dtype=np.int)


def load_merge_set(cache=True):
    def df_add_col_num(df, zfill_width=None):
        if zfill_width is None:
            zfill_width = 2

        mapping = {}
        for idx, key in enumerate(df.keys()):
            mapping[key] = f'col_{str(idx).zfill(zfill_width)}_{key}'

        return df.rename(mapping, axis='columns')

    path = os.getcwd()
    merge_set_path = os.path.join(path, "data", "titanic", "merge_set.csv")
    if not os.path.exists(merge_set_path) or not cache:
        path = os.getcwd()
        train_path = os.path.join(path, "data", "titanic", "train.csv")
        test_path = os.path.join(path, "data", "titanic", "test.csv")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        merged_df = pd.concat([train_df, test_df], axis=0)
        merged_df = df_add_col_num(merged_df)

        merged_df.to_csv(merge_set_path, index=False)
    else:
        merged_df = pd.read_csv(merge_set_path)

    return merged_df


class titanic_null_cleaner(Base_dfCleaner):
    def col_00_Age(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        # print(df.info())

        df[key] = df[key].fillna(999)
        # df = self.fill_random_value_cate(df, key)

        return df

    def col_01_Cabin(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        df[key] = df[key].fillna('none')
        return df

    def col_02_Embarked(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        df = self.fill_major_value_cate(df, key)
        return df

    def col_03_Fare(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        df = self.fill_major_value_cate(df, key)
        return df


class titanic_typecasting(base_df_typecasting):
    def col_00_Age(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_01_Cabin(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print(series.value_counts())
        pass
        return df

    def col_02_Embarked(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_03_Fare(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        df = self.to_float(df, col_key)
        return df

    def col_04_Name(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # print(series.value_counts())
        return df

    def col_05_Parch(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        df = self.to_int(df, col_key)
        return df

    def col_07_Pclass(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_08_Sex(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_09_SibSp(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        df = self.to_int(df, col_key)
        return df

    def col_10_Survived(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # df = self.to_int(df, col_key)
        return df

    def col_11_Ticket(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df


class titanic_transformer(Base_df_transformer):
    def col_00_Age(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # bins = [0, 1, 3, 6, 12, 15, 19, 30, 40, 50, 60, 80 + 1, 1000]

        bins = [0, 15, 19, 30, 40, 50, 60, 80 + 1, 1000]
        # print(series.value_counts())
        binning_df = self.binning(df, col_key, bins)

        col_key_binning = col_key + '_binning'

        binning_df[binning_df[col_key_binning] == 'bin7_[81~1000)'] = 'bin8_missing'
        # print(binning_df[col_key_binning].value_counts())
        df = self.df_concat(df, binning_df)

        encoded_df = self.fetools.to_label(binning_df, col_key + '_binning')
        df = self.df_concat(df, encoded_df)

        return df

    def col_01_Cabin(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        col_key = 'col_01_Cabin_head'
        cabin_head = pd.DataFrame({
            'col_01_Cabin_head': series
        })

        head_alphabet = cabin_head.query(f'{col_key} != "none"')
        head_alphabet = head_alphabet[col_key].astype(str)
        head_alphabet = head_alphabet.str.slice(0, 1)
        cabin_head.loc[head_alphabet.index, col_key] = head_alphabet

        head_na = cabin_head.query(f'{col_key}.isna()')
        cabin_head.loc[head_na.index, col_key] = 'None'
        df = self.df_concat(df, cabin_head)

        encoded_df = self.fetools.to_label(cabin_head, col_key)
        df = self.df_concat(df, encoded_df)

        return df

    def col_02_Embarked(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        encoded_df = self.fetools.to_label(df, col_key)
        df = self.df_concat(df, encoded_df)

        return df

    def col_03_Fare(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        # bins = [0, 5, 10, 20, 30, 120, 6000]
        bins = [0, 10.9, 74.7, 514]

        # val = np.array(list(df[col_key].astype(float)))
        # log_scale_df = DF({
        #     'log_scale': np.log(val + 1),
        #     Ys_key:      df[Ys_key],
        #     'origin':    series
        # })
        # print(log_scale_df.corr())
        #
        # col_log_scale = 'log_scale'
        # log_scale_df = DF({
        #     'log_scale': np.log(val + 1),
        #     # Ys_key:      df[Ys_key],
        #     # 'origin': series
        # })
        # df = self.df_concat(df, log_scale_df)
        # print(log_scale_df.describe())

        binning_df = self.binning(df, col_key, bins)
        df = self.df_concat(df, binning_df)
        encode_df = self.fetools.to_label(df, col_key + '_binning')
        df = self.df_concat(df, encode_df)

        return df

    def col_04_Name(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        data = series

        # split full_name and alias_name
        data = data.str.split("(", expand=True)
        full_name = data[0]

        # split full_name to first honorific last
        full_name = full_name.str.split(",", expand=True)
        first_name = pd.Series(full_name[0])

        full_name = full_name[1].str.split(".", expand=True)
        Honorific = full_name[0].str.replace(' ', '')
        last_name = full_name[1]

        name_df = pd.DataFrame({
            col_key + "_first_name": first_name.astype(str),
            col_key + "_Honorific": Honorific.astype(str),
            col_key + "_last_name": last_name.astype(str)
        })
        df = self.df_concat(df, name_df)

        col_Honorific = 'col_04_Name_Honorific'
        Honorific = DF(df[col_Honorific])
        Honorific_rare_values = [
            'theCountess', 'Sir', 'Ms', 'Mme', 'Mlle', 'Lady',
            'Capt', 'Don', 'Jonkheer', 'Rev', 'Dona',
            'Col', 'Dr', 'Major'
        ]
        Honorific.loc[Honorific[col_Honorific].isin(Honorific_rare_values), col_Honorific] = 'rare'

        col_Honorific_binned = 'col_04_Name_Honorific_binned'
        Honorific_binned = DF({
            col_Honorific_binned: Honorific[col_Honorific]
        })
        # print(df.info())
        # print(Honorific_binned[col_Honorific_binned].value_counts())
        df = self.df_concat(df, Honorific_binned)
        # print(df.info())

        encoded_df = self.fetools.to_label(Honorific_binned, col_Honorific_binned)
        df = self.df_concat(df, encoded_df)
        return df

    def col_05_Parch(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_06_PassengerId(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_07_Pclass(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_08_Sex(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        encoded_df = self.fetools.to_label(df, col_key)
        df = self.df_concat(df, encoded_df)
        return df

    def col_09_SibSp(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def col_11_Ticket(self, df: DF, col_key: str, partial_df: DF, series: Series, Xs_key: list, Ys_key: list):
        def split_ticket(df):
            part = df[col_key].str.split(" ")
            ticket_head = []
            ticket_num = []
            for item in part:
                if len(item) == 1:
                    ticket_head.append("None")
                    ticket_num.append(item[0])
                else:
                    ticket_head.append(item[0])
                    ticket_num.append(item[-1])

            ticket_head_df = pd.DataFrame({
                "col_11_Ticket_head": ticket_head
            }, dtype=str)
            ticket_num_df = pd.DataFrame({
                "col_11_Ticket_num": ticket_num
            }, dtype=str)
            return ticket_head_df, ticket_num_df

        def trans_ticket_number(df, ticket_number):
            # merge_set_df[SURVIVED] = pd.to_numeric(merge_set_df[SURVIVED], downcast='float')
            # merge_set_df = pd.concat([merge_set_df, merge_ticket_head], axis=1)

            # data = df
            col_ticket_number = 'col_11_Ticket_num'

            data = pd.concat([ticket_number, df[[Ys_key]]], axis=1)

            groupby = data.groupby(col_ticket_number)[Ys_key]
            groupby = groupby.agg(['mean'])
            groupby = groupby.sort_values('mean')

            def get_idx(data, query):
                return list(data.query(query).index)

            idx_survived_0 = get_idx(groupby, "mean == 0.0")
            data.loc[data[col_ticket_number].isin(idx_survived_0), [col_ticket_number]] = 'ticket_head_survived_0'

            idx_survived_0_20 = get_idx(groupby, f'0.0 < mean <= 0.2')
            data.loc[data[col_ticket_number].isin(idx_survived_0_20), [col_ticket_number]] = 'ticket_head_survived_0_20'

            idx_survived_30_40 = get_idx(groupby, f'0.3 < mean < 0.4')
            data.loc[
                data[col_ticket_number].isin(idx_survived_30_40), [col_ticket_number]] = 'ticket_head_survived_30_40'

            idx_survived_40_50 = get_idx(groupby, f'0.4 <= mean < 0.5')
            data.loc[
                data[col_ticket_number].isin(idx_survived_40_50), [col_ticket_number]] = 'ticket_head_survived_40_50'

            idx_survived_50_60 = get_idx(groupby, f'0.5 <= mean <= 0.6')
            data.loc[
                data[col_ticket_number].isin(idx_survived_50_60), [col_ticket_number]] = 'ticket_head_survived_50_60'

            idx_survived_60_80 = get_idx(groupby, f'0.6 < mean <= 0.8')
            data.loc[
                data[col_ticket_number].isin(idx_survived_60_80), [col_ticket_number]] = 'ticket_head_survived_60_80'

            idx_survived_80_100 = get_idx(groupby, f'0.8 < mean ')
            data.loc[
                data[col_ticket_number].isin(idx_survived_80_100), [col_ticket_number]] = 'ticket_head_survived_80_100'

            idx_survived_nan = groupby[groupby['mean'].isna()]
            idx_survived_nan = list(idx_survived_nan.index)
            data.loc[data[col_ticket_number].isin(idx_survived_nan), [col_ticket_number]] = 'ticket_head_survived_nan'

            return pd.DataFrame({
                col_ticket_number: data[col_ticket_number]
            })

        ticket_head_df, ticket_num_df = split_ticket(partial_df)
        df = self.df_concat(df, ticket_head_df)

        ticket_num_df = trans_ticket_number(df, ticket_num_df)
        df = self.df_concat(df, ticket_num_df)

        return df

    def col_new_0_family_size(self, df: DF, Xs_key: list, Ys_key: list):
        col = 'col_12_family_size'
        family_size_df = DF({
            col: df['col_09_SibSp'] + df['col_05_Parch'] + 1
        })
        df = self.df_concat(df, family_size_df)

        bins = [1, 2, 5, 12]
        binned = self.binning(family_size_df, col, bins)
        df = self.df_concat(df, binned)

        encoded_df = self.fetools.to_label(binned, col + '_binning')
        df = self.df_concat(df, encoded_df)

        return df

    def col_new_1_party_size(self, df: DF, Xs_key: list, Ys_key: list):
        col = 'col_13_party_size'

        col_ticket = 'col_11_Ticket'
        col_id = 'col_06_PassengerId'
        groupby_df = DF(df.groupby([col_ticket])[col_id].count())
        groupby_df[col_ticket] = groupby_df.index
        groupby_df[col] = groupby_df[col_id]
        groupby_df = groupby_df.drop(columns=col_id)
        # groupby_df = groupby_df.reset_index(drop=True)

        partial = df[[col_ticket, col_id]]
        merged = pd.merge(groupby_df, partial, on=[col_ticket])
        merged = merged.drop(columns=col_ticket)
        merged = merged.sort_values(by=col_id)
        merged = merged.reset_index(drop=True)

        roommate_size_df = DF({
            col: merged[col]
        })
        df = self.df_concat(df, roommate_size_df)

        # bins = [1, 2, 5, 12]
        # binned = self.binning(roommate_size_df, col, bins)
        # df = self.df_concat(df, binned)
        #
        # encoded_df = self.LabelEncoder(binned, col + '_binning')
        # df = self.df_concat(df, encoded_df)

        bins = [1, 2, 5, 12]
        binned = self.binning(roommate_size_df, col, bins)
        df = self.df_concat(df, binned)

        encoded_df = self.fetools.to_label(binned, col + '_binning')
        df = self.df_concat(df, encoded_df)

        return df

    def col_new_2_group_first_name_count(self, df: DF, Xs_key: list, Ys_key: list):
        col = 'col_14_group_first_name_count'

        col_ticket = 'col_11_Ticket'
        col_first_name = 'col_04_Name_first_name'
        groupby = df.groupby([col_ticket])[col_first_name].value_counts()
        groupby_df = DF(groupby)
        # print(groupby_df)

        groupby_df[col_ticket] = groupby_df.index
        groupby_df = groupby_df.reset_index(drop=True)
        groupby_df[col] = groupby_df[col_first_name]
        groupby_df[[col_ticket, col_first_name]] = groupby_df[col_ticket].apply(pd.Series)

        col_id = 'col_06_PassengerId'
        partial = df[[col_first_name, col_ticket, col_id]]
        merged_df = pd.merge(partial, groupby_df, on=[col_first_name, col_ticket])
        merged_df = merged_df.drop(columns=[col_first_name, col_ticket])
        merged_df = merged_df.sort_values(col_id)
        merged_df = merged_df.reset_index(drop=True)
        first_name_count = merged_df.drop(columns=[col_id])
        first_name_count = DF({
            col: first_name_count[col]
        })

        # print(first_name_count)

        df = pd.concat([df, first_name_count], axis=1)

        return df

    def col_new_3_trans_with_only_family(self, df: DF, Xs_key: list, Ys_key: list):
        col = 'col_15_with_only_family'
        with_not_only_family = pd.DataFrame({
            col: [0] * len(df)
        })
        idxs = df.query(f""" col_13_party_size != col_14_group_first_name_count""").index
        with_not_only_family.loc[idxs, col] = 1

        df = self.df_concat(df, with_not_only_family)

        return df


def split_train_test(df):
    test = df.query('col_10_Survived.isnull()')
    test = test.drop(columns=['col_10_Survived'])
    train = df.query('not col_10_Survived.isnull()')
    return train, test


def build_dataset(path):
    merge_df = load_merge_set()

    cleaner = titanic_null_cleaner(merge_df, df_Xs_keys, df_Ys_key)
    cleaner.boilerplate_maker('./titanic_cleaner.py')
    merge_df = cleaner.clean()

    typecaster = titanic_typecasting(merge_df, df_Xs_keys, df_Ys_key)
    typecaster.boilerplate_maker('./titanic_typecaster.py')
    merge_df = typecaster.type_cast()

    transformer = titanic_transformer(merge_df, df_Xs_keys, df_Ys_key)
    transformer.boilerplate_maker('./titanic_transformer.py')
    merge_df = transformer.transform()

    train, test = split_train_test(merge_df)

    train.to_csv(path_join(path, 'trans_train.csv'), index=False)
    test.to_csv(path_join(path, 'trans_test.csv'), index=False)


class titanic_train(BaseDataset):
    FILE_NAME = "train.csv"

    def load(self, path):
        trans_path = os.path.join(path, "trans_train.csv")

        if not self.caching or not os.path.exists(trans_path):
            build_dataset(path)

        df = pd.read_csv(trans_path, index_col=None)

        Xs_col = [
            'col_00_Age_binning_encoded',
            'col_01_Cabin_head_encoded',
            'col_02_Embarked_encoded',
            'col_03_Fare_binning_encoded',
            'col_04_Name_Honorific_binned_encoded',
            'col_05_Parch',
            'col_07_Pclass',
            'col_08_Sex_encoded',
            'col_09_SibSp',
            'col_12_family_size_binning_encoded',
            'col_13_party_size_binning_encoded',
            'col_14_group_first_name_count',
            'col_15_with_only_family'
        ]
        Xs_df = df[Xs_col]
        onehot_np_arr = df_to_np_onehot_embedding(Xs_df)
        self.add_data('x', onehot_np_arr)
        self.x_keys = Xs_col

        Ys_df = df[['col_10_Survived']]
        Ys_df = df_to_onehot_embedding(Ys_df)
        self.add_data('y', np.array(Ys_df))
        self.y_keys = ['col_10_Survived']


class titanic_test(BaseDataset):
    FILE_NAME = "test.csv"

    def load(self, path):
        trans_path = os.path.join(path, "trans_test.csv")

        if not self.caching or not os.path.exists(trans_path):
            build_dataset(path)

        df = pd.read_csv(trans_path, index_col=False)

        Xs_col = [
            'col_00_Age_binning_encoded',
            'col_01_Cabin_head_encoded',
            'col_02_Embarked_encoded',
            'col_03_Fare_binning_encoded',
            'col_04_Name_Honorific_binned_encoded',
            'col_05_Parch',
            'col_07_Pclass',
            'col_08_Sex_encoded',
            'col_09_SibSp',
            'col_12_family_size_binning_encoded',
            'col_13_party_size_binning_encoded',
            'col_14_group_first_name_count',
            'col_15_with_only_family'
        ]
        Xs_df = df[Xs_col]
        onehot_np_arr = df_to_np_onehot_embedding(Xs_df)
        self.add_data('x', onehot_np_arr)
        self.x_keys = Xs_col


class titanic(BaseDatasetPack):
    LABEL_SIZE = 2

    def __init__(self, caching=True, verbose=20, **kwargs):
        super().__init__(caching, verbose=verbose, **kwargs)

        self.train_set = titanic_train(caching=caching, verbose=verbose)
        self.test_set = titanic_test(caching=caching, verbose=verbose)
        self.pack['train'] = self.train_set
        self.pack['test'] = self.test_set

    @staticmethod
    def to_kaggle_submit_csv(path, Ys):
        if path is None:
            path = path_join('.', 'submit.csv')
        df = pd.DataFrame()

        df[PASSENGERID] = [i for i in range(892, 1309 + 1)]
        df[SURVIVED] = Ys

        df.to_csv(path, index=False)
