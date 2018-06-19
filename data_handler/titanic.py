from pprint import pprint

from util.misc_util import path_join
from util.numpy_utils import *
from data_handler.BaseDataset import BaseDataset
from data_handler.BaseDatasetPack import BaseDatasetPack
import os
import pandas as pd


def np_str_labels_to_index(np_arr, labels):
    np_arr = np.asarray(np_arr)
    new_arr = np.zeros_like(np_arr)
    for idx, label in enumerate(labels):
        new_arr = np.where(np_arr == label, idx, new_arr)

    return new_arr.astype(dtype=np.int)


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


# def pprint_info_pack(df):
#     pprint(df.info())
#     for key in df.keys():
#         pprint(df[key].value_counts())
# def trans_cabbin(df):
#     return df
#
#
# def trans_has_child(df):
#     return df
#
# def trans_is_alone(df):
#     return df


def df_to_np_onehot_embedding(df):
    np_arr = np.array(df)
    ret = {}
    for df_key in df.keys():
        for unique_key in df[df_key].unique():
            ret[unique_key] = np.where(np_arr == unique_key, 1, 0).reshape([-1, 1])

    return ret


def df_bucketize(df, key, bucket_range, column='bucket', na='None', null='None'):
    new_df = pd.DataFrame({column: df[key]})

    for i in range(len(bucket_range) - 1):
        a, b = bucket_range[i: i + 2]
        name = f'{key}_range_{a}~{b}'

        query = f'{a} <= {key} < {b}'

        idx = list(df.query(query).index.values)
        new_df.loc[idx] = name

    idx = list(df.query(f'{key}.isna()').index.values)
    new_df.loc[idx] = na

    idx = list(df.query(f'{key}.isnull()').index.values)
    new_df.loc[idx] = null

    return new_df


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


def trans_fare(self):
    df = self.to_DataFrame([FARE])
    df = df.astype(float)
    bucket_range = [0, 10, 20, 30, 120, 6000]
    df = df_bucketize(df, 'Fare', bucket_range, 'Fare_bucket')

    np_arr = np.array(df)
    ret = {}
    for key in df['Fare_bucket'].unique():
        ret[key] = np.where(np_arr == key, 1, 0).reshape([-1, 1])
    return ret


def trans_age(self):
    df = self.to_DataFrame([AGE])
    df = df.astype(float)
    bucket_range = [0, 1, 3, 6, 12, 15, 19, 30, 40, 50, 60, 80 + 1]
    df = df_bucketize(df, 'Age', bucket_range, 'Age_bucket')

    np_arr = np.array(df)
    ret = {}
    for key in df['Age_bucket'].unique():
        ret[key] = np.where(np_arr == key, 1, 0).reshape([-1, 1])

    return ret


def trans_ticket(self):
    def split_ticket_head_num(ticket_df):
        part = ticket_df["Ticket"].str.split(" ")
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
        merge_ticket_head, _ = split_ticket_head_num(merge_set_df)
        merge_set_df = pd.concat([merge_set_df, merge_ticket_head], axis=1)

        data = merge_set_df
        data = data.groupby('ticket_head')['Survived']
        data = data.agg(['mean'])
        data = data.sort_values('mean')
        ticket_head_per_survived = data

        def get_idx(df, query):
            return list(df.query(query).index)

        idx_survived_0 = get_idx(ticket_head_per_survived, "mean == 0.0")
        df.loc[df['ticket_head'].isin(idx_survived_0), ['ticket_head']] = 'ticket_head_survived_0'

        idx_survived_0_20 = get_idx(ticket_head_per_survived, f'0.0 < mean <= 0.2')
        df.loc[df['ticket_head'].isin(idx_survived_0_20), ['ticket_head']] = 'ticket_head_survived_0_20'

        idx_survived_30_40 = get_idx(ticket_head_per_survived, f'0.3 < mean < 0.4')
        df.loc[df['ticket_head'].isin(idx_survived_30_40), ['ticket_head']] = 'ticket_head_survived_30_40'

        idx_survived_40_50 = get_idx(ticket_head_per_survived, f'0.4 <= mean < 0.5')
        df.loc[df['ticket_head'].isin(idx_survived_40_50), ['ticket_head']] = 'ticket_head_survived_40_50'

        idx_survived_50_60 = get_idx(ticket_head_per_survived, f'0.5 <= mean <= 0.6')
        df.loc[df['ticket_head'].isin(idx_survived_50_60), ['ticket_head']] = 'ticket_head_survived_50_60'

        idx_survived_60_80 = get_idx(ticket_head_per_survived, f'0.6 < mean <= 0.8')
        df.loc[df['ticket_head'].isin(idx_survived_60_80), ['ticket_head']] = 'ticket_head_survived_60_80'

        idx_survived_80_100 = get_idx(ticket_head_per_survived, f'0.8 < mean ')
        df.loc[df['ticket_head'].isin(idx_survived_80_100), ['ticket_head']] = 'ticket_head_survived_80_100'

        idx_survived_nan = ticket_head_per_survived[ticket_head_per_survived['mean'].isna()]
        idx_survived_nan = list(idx_survived_nan.index)
        df.loc[df['ticket_head'].isin(idx_survived_nan), ['ticket_head']] = 'ticket_head_survived_nan'

        return df

    df = self.to_DataFrame()
    ticket_head_df, ticket_num_df = split_ticket_head_num(df)

    ticket_head_df = trans_ticket_head(ticket_head_df)
    ticket_head = df_to_np_onehot_embedding(ticket_head_df)

    ticket_num = np.array(ticket_num_df)

    return ticket_head, ticket_num


def df_to_np_dict(df, dtype=None):
    ret = {}
    for key in df.keys():
        ret[key] = np.array(df[key], dtype=dtype)
    return ret


def trans_name(self):
    def split_name(df):
        data = df['Name']

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
        # most survived
        merge_most_survived = ['theCountess', 'Sir', 'Ms', 'Mme', 'Mlle', 'Lady']

        # most died
        merge_most_died = ['Capt', 'Don', 'Jonkheer', 'Rev', 'Dona']

        # half died
        merge_half_died = ['Col', 'Dr', 'major']

        df.loc[df["Honorific"].isin(merge_most_survived), ['Honorific']] = 'Honorific_survived'
        df.loc[df["Honorific"].isin(merge_most_died), ['Honorific']] = 'Honorific_most_died'
        df.loc[df["Honorific"].isin(merge_half_died), ['Honorific']] = 'Honorific_half_died'

        return df

    df = self.to_DataFrame([NAME])
    first_name_df, honorific_df, last_name_df = split_name(df)

    first_name = df_to_np_dict(first_name_df, dtype=str)

    honorific_df = trans_Honorific(honorific_df)
    honorific = df_to_np_onehot_embedding(honorific_df)

    last_name = df_to_np_dict(last_name_df, dtype=str)

    return first_name, honorific, last_name


def trans_sex(self):
    ret = {}
    data = self.data[SEX]
    ret["Sex_male"] = np.where(data == "male", 1, 0).reshape([-1, 1])
    ret["Sex_female"] = np.where(data == "female", 1, 0).reshape([-1, 1])
    return ret


def trans_embarked(self):
    ret = {}
    data = self.data[EMBARKED]
    ret["Embarked_C"] = np.where(data == "C", 1, 0).reshape([-1, 1])
    ret["Embarked_S"] = np.where(data == "S", 1, 0).reshape([-1, 1])
    ret["Embarked_Q"] = np.where(data == "Q", 1, 0).reshape([-1, 1])
    ret["Embarked_nan"] = np.where(data == "nan", 1, 0).reshape([-1, 1])
    return ret


def trans_pclass(self):
    ret = {}
    data = self.data[PCLASS]
    data = data.astype(np.int)
    data = np_index_to_onehot(data)
    ret["pclass_onehot"] = data
    return ret


def trans_sibsp(self):
    ret = {}
    data = self.data[SIBSP]
    data = data.astype(np.int)
    data = np_index_to_onehot(data)
    ret["sibsp_onehot"] = data
    return ret


def trans_parch(self):
    ret = {}
    data = self.data[PARCH]
    data = data.astype(np.int)
    data = np_index_to_onehot(data, n=10)
    ret["parch_onehot"] = data
    return ret


def trans_family_size(self):
    ret = {}
    sibsp = self.data[SIBSP].astype(np.int)
    parch = self.data[PARCH].astype(np.int)
    ret["family_size_onehot"] = np_index_to_onehot(sibsp + parch + 1, n=20)
    return ret


def x_preprocess(self):
    # df = self.to_DataFrame()
    # fare = trans_fare(df)
    # age = trans_age(df)
    # name = trans_name(df)
    # ticket = trans_ticket(df)
    #
    # merge_df = pd.concat([df, fare, age, name, ticket], axis=1)
    # merge_df = merge_df.drop(columns=['Name', 'Ticket', 'Fare', 'Age'])

    x_dict = {}
    data = self.data[PASSENGERID]
    data = data.astype(np.int)
    self.data[PASSENGERID] = data

    # df = self.to_DataFrame()

    fare = trans_fare(self)
    age = trans_age(self)
    first_name, honorific, last_name = trans_name(self)
    ticket_head, ticket_num = trans_ticket(self)
    sex = trans_sex(self)
    embarked = trans_embarked(self)
    pclass = trans_pclass(self)
    sibsp = trans_sibsp(self)
    parch = trans_parch(self)
    family_size = trans_family_size(self)

    x_dict.update(sex)
    x_dict.update(embarked)
    x_dict.update(pclass)
    x_dict.update(sibsp)
    x_dict.update(parch)
    x_dict.update(family_size)
    x_dict.update(fare)
    x_dict.update(age)
    x_dict.update(honorific)
    x_dict.update(ticket_head)
    # pprint(x_dict.keys())
    return np.concatenate(list(x_dict.values()), axis=1)


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
        data_path = os.path.join(path, self.FILE_NAME)
        pd_data = pd.read_csv(
            data_path,
            sep=',',
            header=None,
            error_bad_lines=False,
            names=self.CSV_COLUMNS,
        )
        pd_data = pd_data.fillna("nan")
        for col, key in zip(self.CSV_COLUMNS, self.BATCH_KEYS):
            self.data[key] = np.array(pd_data[col])[1:]

    def save(self):
        pass

    def transform(self):
        data = x_preprocess(self)
        self.add_data(Xs, data)

        # add train_label
        data = self.data[SURVIVED]
        data = data.astype(np.int)
        data = np_index_to_onehot(data)
        self.add_data(Ys, data)


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
        data_path = os.path.join(path, self.FILE_NAME)
        pd_data = pd.read_csv(
            data_path,
            sep=',',
            header=None,
            error_bad_lines=False,
            names=self.CSV_COLUMNS,
        )
        pd_data = pd_data.fillna('nan')
        for col, key in zip(self.CSV_COLUMNS, self.BATCH_KEYS):
            self.data[key] = np.array(pd_data[col])[1:]

    def save(self):
        pass

    def transform(self):
        data = x_preprocess(self)
        self.add_data(Xs, data)


class titanic(BaseDatasetPack):
    LABEL_SIZE = 2

    def __init__(self):
        super().__init__()
        self.train_set = titanic_train()
        self.test_set = titanic_test()
        self.set['train'] = self.train_set
        self.set['test'] = self.test_set

    def to_kaggle_submit_csv(self, path, Ys):
        if path is None:
            path = path_join('.', 'submit.csv')
        df = pd.DataFrame()

        df[PASSENGERID] = [i for i in range(892, 1309 + 1)]
        df[SURVIVED] = Ys

        pprint(df.head())
        df.to_csv(path, index=False)
