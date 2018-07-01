from script.util.misc_util import path_join
from script.util.numpy_utils import *
from script.data_handler.BaseDataset import BaseDataset
from script.data_handler.BaseDatasetPack import BaseDatasetPack
import os
import pandas as pd
from script.util.pandas_util import df_bucketize, df_to_np_dict, df_to_onehot_embedding

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


# def trans_has_child(df):
#     return df
#

def np_str_labels_to_index(np_arr, labels):
    np_arr = np.asarray(np_arr)
    new_arr = np.zeros_like(np_arr)
    for idx, label in enumerate(labels):
        new_arr = np.where(np_arr == label, idx, new_arr)

    return new_arr.astype(dtype=np.int)


def load_merge_set():
    path = os.getcwd()
    merge_set_path = os.path.join(path, "data", "titanic", "merge_set.csv")
    if not os.path.exists(merge_set_path):
        path = os.getcwd()
        train_path = os.path.join(path, "data", "titanic", "train.csv")
        test_path = os.path.join(path, "data", "titanic", "test.csv")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        merged_df = pd.concat([train_df, test_df], axis=1)
        merged_df = merged_df.reset_index(drop=True)

        merged_df.to_csv(merge_set_path)
    else:
        merged_df = pd.read_csv(merge_set_path)

    return merged_df


def trans_fare(df):
    df = df[[FARE]]
    df = df.astype(float)

    bucket_range = [0, 10, 20, 30, 120, 6000]
    df = df_bucketize(df, 'Fare', bucket_range, 'Fare_bucket')

    return df


def trans_age(df):
    df = df[[AGE]]
    df = df.astype(float)
    bucket_range = [0, 1, 3, 6, 12, 15, 19, 30, 40, 50, 60, 80 + 1]
    df = df_bucketize(df, 'Age', bucket_range, 'Age_bucket')

    return df


def split_ticket(df):
    part = df["Ticket"].str.split(" ")
    ticket_head = []
    ticket_num = []
    for item in part:
        if len(item) == 1:
            ticket_head.append("None")
            ticket_num.append(item[0])
        else:
            ticket_head.append(item[0])
            ticket_num.append(item[-1])

    ticket_head = pd.DataFrame({"ticket_head": ticket_head}, dtype=str)
    ticket_num = pd.DataFrame({"ticket_num": ticket_num}, dtype=str)
    return ticket_head, ticket_num


def trans_ticket_head(df):
    merge_set_df = load_merge_set()
    merge_set_df[SURVIVED] = pd.to_numeric(merge_set_df[SURVIVED], downcast='float')
    merge_ticket_head, _ = split_ticket(merge_set_df)
    merge_set_df = pd.concat([merge_set_df, merge_ticket_head], axis=1)

    data = merge_set_df
    data = data.groupby('ticket_head')['Survived']
    data = data.agg(['mean'])
    data = data.sort_values('mean')
    groupby = data

    def get_idx(df, query):
        return list(df.query(query).index)

    idx_survived_0 = get_idx(groupby, "mean == 0.0")
    df.loc[df['ticket_head'].isin(idx_survived_0), ['ticket_head']] = 'ticket_head_survived_0'

    idx_survived_0_20 = get_idx(groupby, f'0.0 < mean <= 0.2')
    df.loc[df['ticket_head'].isin(idx_survived_0_20), ['ticket_head']] = 'ticket_head_survived_0_20'

    idx_survived_30_40 = get_idx(groupby, f'0.3 < mean < 0.4')
    df.loc[df['ticket_head'].isin(idx_survived_30_40), ['ticket_head']] = 'ticket_head_survived_30_40'

    idx_survived_40_50 = get_idx(groupby, f'0.4 <= mean < 0.5')
    df.loc[df['ticket_head'].isin(idx_survived_40_50), ['ticket_head']] = 'ticket_head_survived_40_50'

    idx_survived_50_60 = get_idx(groupby, f'0.5 <= mean <= 0.6')
    df.loc[df['ticket_head'].isin(idx_survived_50_60), ['ticket_head']] = 'ticket_head_survived_50_60'

    idx_survived_60_80 = get_idx(groupby, f'0.6 < mean <= 0.8')
    df.loc[df['ticket_head'].isin(idx_survived_60_80), ['ticket_head']] = 'ticket_head_survived_60_80'

    idx_survived_80_100 = get_idx(groupby, f'0.8 < mean ')
    df.loc[df['ticket_head'].isin(idx_survived_80_100), ['ticket_head']] = 'ticket_head_survived_80_100'

    idx_survived_nan = groupby[groupby['mean'].isna()]
    idx_survived_nan = list(idx_survived_nan.index)
    df.loc[df['ticket_head'].isin(idx_survived_nan), ['ticket_head']] = 'ticket_head_survived_nan'

    return pd.DataFrame({'ticket_head': df['ticket_head']})


def split_name(df):
    data = df[NAME]

    # split full_name and alias_name
    data = data.str.split("(", expand=True)
    full_name = data[0]

    # split full_name to first honorific last
    full_name = full_name.str.split(",", expand=True)
    first_name = pd.Series(full_name[0])

    full_name = full_name[1].str.split(".", expand=True)
    Honorific = full_name[0].str.replace(' ', '')
    last_name = full_name[1]

    first_name = pd.DataFrame({
        "first_name": first_name.astype(str),
    })

    Honorific = pd.DataFrame({
        "Honorific": Honorific.astype(str),
    })

    last_name = pd.DataFrame({
        "last_name": last_name.astype(str)
    })

    return first_name, Honorific, last_name


def trans_Honorific(df):
    df = df[['Honorific']]

    # most survived
    merge_most_survived = ['theCountess', 'Sir', 'Ms', 'Mme', 'Mlle', 'Lady']

    # most died
    merge_most_died = ['Capt', 'Don', 'Jonkheer', 'Rev', 'Dona']

    # half died
    merge_half_died = ['Col', 'Dr', 'major']

    df.loc[df["Honorific"].isin(merge_most_survived), ['Honorific']] = 'Honorific_survived'
    df.loc[df["Honorific"].isin(merge_most_died), ['Honorific']] = 'Honorific_most_died'
    df.loc[df["Honorific"].isin(merge_half_died), ['Honorific']] = 'Honorific_half_died'

    return pd.DataFrame({'Honorific': df['Honorific'].values})


def trans_sex(df):
    return pd.DataFrame({SEX: df[SEX].values})


def trans_embarked(df):
    idxs = list(df.query('Embarked.isna()').index.values)
    df.loc[idxs, [EMBARKED]] = "C"

    return pd.DataFrame({EMBARKED: df[EMBARKED].values})


def trans_pclass(df):
    df = df[[PCLASS]]
    df = df.astype(int)

    return pd.DataFrame({PCLASS: df[PCLASS].values})


def trans_sibsp(df):
    df = df[[SIBSP]]
    df = df.astype(np.int)
    return pd.DataFrame({SIBSP: df[SIBSP].values})


def trans_parch(df):
    df = df[[PARCH]]
    df = df.astype(np.int)
    return pd.DataFrame({PARCH: df[PARCH].values})


def trans_family_size(df):
    return pd.DataFrame({'family_size': df[SIBSP] + df[PARCH] + 1})


def trans_room_mate_number(ticket_df, passangerId_df):
    df = pd.concat([ticket_df, passangerId_df], axis=1)
    groupby = df.groupby('Ticket')['PassengerId'].count()

    ret = pd.DataFrame({'room_mate_number': [0] * len(df)})
    for ticket in list(groupby.index.values):
        ret.loc[df['Ticket'] == ticket, ['room_mate_number']] = groupby[ticket]

    return ret


def trans_group_first_name_count(ticket_df, first_name_df):
    df = pd.concat([ticket_df, first_name_df], axis=1)

    groupby = df.groupby(['Ticket'])['first_name'].value_counts()

    ret = pd.DataFrame({'group_first_name_count': [0] * len(df)})
    for ticket, first_name in list(groupby.index.values):
        idxs = df.query(f"""Ticket == "{ticket}" and first_name == "{first_name}" """).index
        ret.loc[idxs, 'group_first_name_count'] = groupby[ticket, first_name]

    return ret


def trans_with_not_only_family(room_mate_number_df, group_first_name_count_df):
    df = pd.concat([room_mate_number_df, group_first_name_count_df], axis=1)

    ret = pd.DataFrame({'with_not_only_family': [0] * len(df)})
    idxs = df.query(f""" room_mate_number != group_first_name_count""").index
    ret.loc[idxs, 'with_not_only_family'] = 1

    return ret


def split_train_test(df):
    test = df.query('Survived.isnull()')
    test = test.drop(columns=['Survived'])
    train = df.query('not Survived.isnull()')
    return train, test


def trans_cabin(df):
    df = pd.DataFrame(df['Cabin'])

    cabin_head = df.query('not Cabin.isna()')
    cabin_head = cabin_head['Cabin'].astype(str)
    cabin_head = cabin_head.str.slice(0, 1)

    na = df.query('Cabin.isna()')

    df.loc[na.index, 'Cabin'] = 'None'
    df.loc[cabin_head.index, 'Cabin'] = cabin_head
    return df


def build_transform(df):
    fare = trans_fare(df)
    age = trans_age(df)
    first_name, honorific, last_name = split_name(df)
    honorific = trans_Honorific(honorific)
    ticket_head, ticket_num = split_ticket(df)
    ticket_head = trans_ticket_head(ticket_head)
    sex = trans_sex(df)
    embarked = trans_embarked(df)
    pclass = trans_pclass(df)
    sibsp = trans_sibsp(df)
    parch = trans_parch(df)
    cabin = trans_cabin(df)

    room_mate_number = trans_room_mate_number(df[[TICKET]], df[[PASSENGERID]])
    family_size = trans_family_size(df)
    group_first_name_count = trans_group_first_name_count(df[[TICKET]], first_name)
    with_not_only_family = trans_with_not_only_family(room_mate_number, group_first_name_count)

    x_df = pd.concat(
        [fare,
         age,
         honorific,
         sex,
         embarked,
         pclass,
         sibsp,
         parch,
         family_size,
         ticket_head,
         room_mate_number,
         group_first_name_count,
         with_not_only_family,
         cabin], axis=1)
    return x_df


class titanic_train(BaseDataset):
    BATCH_KEYS = [
        PASSENGERID,
        SURVIVED,
        PCLASS,
        NAME,
        SEX,
        AGE,
        SIBSP,
        PARCH,
        TICKET,
        FARE,
        CABIN,
        EMBARKED,
        Xs,
        Ys
    ]

    CSV_COLUMNS = [
        PASSENGERID,
        SURVIVED,
        PCLASS,
        NAME,
        SEX,
        AGE,
        SIBSP,
        PARCH,
        TICKET,
        FARE,
        CABIN,
        EMBARKED,
    ]

    FILE_NAME = "train.csv"

    def load(self, path, limit=None):
        trans_path = os.path.join(path, "trans_train.csv")

        if not self.caching or not os.path.exists(trans_path):
            df = load_merge_set()
            merge_transform = build_transform(df)
            merge_transform_onehot = df_to_onehot_embedding(merge_transform)
            merge_df = pd.concat([merge_transform_onehot, df[[SURVIVED]]], axis=1)
            train, test = split_train_test(merge_df)

            merge_transform.to_csv(path_join(path, 'trans_merge.csv'))
            train.to_csv(path_join(path, 'trans_train.csv'), index=False)
            test.to_csv(path_join(path, 'trans_test.csv'), index=False)

        df = pd.read_csv(trans_path, index_col=False)
        self.data = df_to_np_dict(df)

    def save(self):
        pass

    def transform(self):
        df = self.to_DataFrame()

        id_ = pd.DataFrame(df.pop('id_'))
        self.add_data('id_', np.array(id_))

        Ys_df = pd.DataFrame(df.pop(SURVIVED))
        Ys_df = Ys_df.astype(int)
        Ys_df = df_to_onehot_embedding(Ys_df)
        self.add_data('Ys', np.array(Ys_df))

        Xs_df = df
        self.add_data('Xs', np.array(Xs_df))


class titanic_test(BaseDataset):
    BATCH_KEYS = [
        PASSENGERID,
        PCLASS,
        NAME,
        SEX,
        AGE,
        SIBSP,
        PARCH,
        TICKET,
        FARE,
        CABIN,
        EMBARKED,
    ]

    CSV_COLUMNS = [
        PASSENGERID,
        PCLASS,
        NAME,
        SEX,
        AGE,
        SIBSP,
        PARCH,
        TICKET,
        FARE,
        CABIN,
        EMBARKED,
    ]

    FILE_NAME = "test.csv"

    def load(self, path, limit=None):
        trans_path = os.path.join(path, "trans_test.csv")

        if not self.caching or not os.path.exists(trans_path):
            df = load_merge_set()
            merge_transform = build_transform(df)
            merge_transform_onehot = df_to_onehot_embedding(merge_transform)
            merge_df = pd.concat([merge_transform_onehot, df[[SURVIVED]]], axis=1)
            train, test = split_train_test(merge_df)

            merge_transform.to_csv(path_join(path, 'trans_merge.csv'))
            train.to_csv(path_join(path, 'trans_train.csv'), index=False)
            test.to_csv(path_join(path, 'trans_test.csv'), index=False)

        df = pd.read_csv(trans_path, index_col=False)
        self.data = df_to_np_dict(df)

    def save(self):
        pass

    def transform(self):
        df = self.to_DataFrame()

        id_ = pd.DataFrame(df.pop('id_'))
        self.add_data('id_', np.array(id_))

        Xs_df = df
        self.add_data('Xs', np.array(Xs_df))


class titanic(BaseDatasetPack):
    LABEL_SIZE = 2

    def __init__(self, caching=True, **kwargs):
        super().__init__(caching, **kwargs)

        self.train_set = titanic_train(caching=caching)
        self.test_set = titanic_test(caching=caching)
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
