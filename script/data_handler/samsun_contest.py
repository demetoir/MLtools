import os
from pprint import pprint
import pandas as pd
from script.data_handler.Base.BaseDataset import BaseDataset
from script.data_handler.Base.BaseDatasetPack import BaseDatasetPack
from script.sklearn_like_toolkit.FETools import FETools, DF_binning_encoder
from script.util.misc_util import path_join
from script.util.numpy_utils import *

DF = pd.DataFrame
Series = pd.Series

origin_csv_path = "./data/dataset_kor/교통사망사고정보/Kor_Train_교통사망사고정보(12.1~17.6).csv"
path_head = "./data/dataset_kor"


def dict_transpose(d):
    return {val: key for key, val in d.items()}


def add_col_num(df, zfill=2):
    df_cols = df.columns
    new_cols = []
    for i, col in enumerate(df_cols):
        new_cols += ['c' + str(i).zfill(zfill) + '_' + col]
    df.columns = new_cols
    return df


def drop_rows(df, row_idxs):
    return df.drop(row_idxs, axis=0)


def load_samsung(path):
    return pd.read_csv(path, encoding='euc-kr', engine='python')


def save_samsung(df, path):
    df.to_csv(path, encoding='euc-kr', index=False)


class transformer:

    def __init__(self):
        self.inverse_funcs = {}
        self.encoders = {}
        self.cols = [
            'c00_주야',
            'c01_요일',
            'c02_사망자수',
            'c03_사상자수',
            'c04_중상자수',
            'c05_경상자수',
            'c06_부상신고자수',
            'c07_발생지시도',
            'c08_발생지시군구',
            'c09_사고유형_대분류',
            'c10_사고유형_중분류',
            'c11_법규위반',
            'c12_도로형태_대분류',
            'c13_도로형태',
            'c14_당사자종별_1당_대분류',
            'c15_당사자종별_2당_대분류'
        ]
        self.trans_funcs = [
            'c_00',
            'c_01',
            'c_02',
            'c_03',
            'c_04',
            'c_05',
            'c_06',
            'c_07',
            'c_08',
            'c_09',
            'c_10',
            'c_11',
            'c_12',
            'c_13',
            'c_14',
            'c_15',
        ]

        self.inverse_funcs = [
            'inv_c_00',
            'inv_c_01',
            'inv_c_02',
            'inv_c_03',
            'inv_c_04',
            'inv_c_05',
            'inv_c_06',
            'inv_c_07',
            'inv_c_08',
            'inv_c_09',
            'inv_c_10',
            'inv_c_11',
            'inv_c_12',
            'inv_c_13',
            'inv_c_14',
            'inv_c_15',
        ]

    def transform(self, df):
        for func_name in self.trans_funcs:
            if hasattr(self, func_name):
                func = getattr(self, func_name)
                df = func(df)

        df = df[list(sorted(df.columns))]
        return df

    def inverse_transform(self, df):
        for idx, func_name in enumerate(self.inverse_funcs):
            col = self.cols[idx]
            if hasattr(self, func_name):
                func = getattr(self, func_name)
                df = func(df, col)

        return df

    def c_15(self, df, key='c15_당사자종별_2당_대분류'):
        df.loc[df[key] == '0', key] = '없음'
        fe_tool = FETools()
        df = fe_tool.drop_row_by_value(df, key, '사륜오토바이(ATV)')
        df = fe_tool.drop_row_by_value(df, key, '열차')

        return df

    def c_14(self, df, key='c14_당사자종별_1당_대분류'):
        fe_tool = FETools()
        df = fe_tool.drop_row_by_value(df, key, '개인형이동수단(PM)')

        return df

    def c_13(self, df, key='c13_도로형태'):
        fe_tool = FETools()
        df = fe_tool.drop_row_by_value(df, key, '지하차도(도로)내')
        df = fe_tool.drop_row_by_value(df, key, '불명')
        df = fe_tool.drop_row_by_value(df, key, '주차장')

        return df

    def c_12(self, df, key='c12_도로형태_대분류'):
        fe_tool = FETools()
        df = fe_tool.drop_row_by_value(df, key, '지하차도(도로)내')
        df = fe_tool.drop_row_by_value(df, key, '불명')
        df = fe_tool.drop_row_by_value(df, key, '주차장')

        return df

    def c_11(self, df, key='c11_법규위반'):
        fe_tool = FETools()
        df = fe_tool.drop_row_by_value(df, key, '통행우선 순위위반')
        df = fe_tool.drop_row_by_value(df, key, '보행자과실')
        df = fe_tool.drop_row_by_value(df, key, '과로')
        df = fe_tool.drop_row_by_value(df, key, '진로양보 의무 불이행')

        return df

    def _binning_encode(self, df, key, bins):
        if key not in self.encoders:
            encoder = DF_binning_encoder()
            encoder.fit(df, key, bins)
            self.encoders[key] = encoder
        else:
            encoder = self.encoders[key]

        binning_df = encoder.encode(df)
        df[key] = binning_df[key]
        return df

    def c_06(self, df, key='c06_부상신고자수'):
        bins = [0, 1, 2, 3, 5, 100]
        df = self._binning_encode(df, key, bins)

        return df

    def c_05(self, df, key='c05_경상자수'):
        bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
        df = self._binning_encode(df, key, bins)

        return df

    def c_04(self, df, key='c04_중상자수'):
        bins = [0, 1, 2, 3, 4, 5, 6, 7, 9, 100]
        df = self._binning_encode(df, key, bins)

        return df

    def c_03(self, df, key='c03_사상자수'):
        bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 999]
        df = self._binning_encode(df, key, bins)

        return df

    def c_02(self, df, key='c02_사망자수'):
        bins = [0, 1, 2, 3, 4, 100]
        df = self._binning_encode(df, key, bins)

        return df

    def inv_c_06(self, df, key):
        encoder = self.encoders[key]
        decoded = encoder.decode(df)
        df[key] = decoded[key]
        return df

    def inv_c_05(self, df, key):
        encoder = self.encoders[key]
        decoded = encoder.decode(df)
        df[key] = decoded[key]
        return df

    def inv_c_04(self, df, key):
        encoder = self.encoders[key]
        decoded = encoder.decode(df)
        df[key] = decoded[key]
        return df

    def inv_c_03(self, df, key):
        encoder = self.encoders[key]
        decoded = encoder.decode(df)
        df[key] = decoded[key]
        return df

    def inv_c_02(self, df, key):
        encoder = self.encoders[key]
        decoded = encoder.decode(df)
        df[key] = decoded[key]
        return df


def init_samsung(cache=True):
    init_csv_path = path_join(path_head, 'data_init.csv')
    if not os.path.exists(init_csv_path) or not cache:
        df = load_samsung(origin_csv_path)
        # train columns to make like test columns
        path = path_join(path_head, 'test_kor.csv')
        test_df = load_samsung(path)
        test_cols = test_df.columns
        df = DF(df[test_cols])
        df = add_col_num(df, 2)

        # drop null include rows
        idx = df[df['c15_당사자종별_2당_대분류'].isna()].index
        df = drop_rows(df, idx)
        df = DF(df)

        save_samsung(df, init_csv_path)
    else:
        df = load_samsung(init_csv_path)

    return df


def samsung_transform(cache=False):
    init_samsung()
    path = path_join(path_head, 'data_init.csv')
    df = load_samsung(path)

    id_col = 'c.._id'
    path = path_join(path_head, 'data_tansformed.csv')
    if not os.path.exists(path) or not cache:
        trans = transformer()
        df = trans.transform(df)
        df = df.drop_duplicates(keep='first')

        df[id_col] = np.arange(0, len(df))
        save_samsung(df, path)

    df = load_samsung(path)
    reg_cols = [
        'c02_사망자수',
        'c03_사상자수',
        'c04_중상자수',
        'c05_경상자수',
        'c06_부상신고자수',
    ]

    label_encoder_cols = []
    for k in df.columns:
        if '_label' in k:
            label_encoder_cols += [k]

    onehot_col = []
    for k in df.columns:
        if '_onehot' in k:
            onehot_col += [k]

    x_cols = reg_cols + onehot_col

    origin_cols = [
        'c00_주야',
        'c01_요일',
        'c02_사망자수',
        'c03_사상자수',
        'c04_중상자수',
        'c05_경상자수',
        'c06_부상신고자수',
        'c07_발생지시도',
        'c08_발생지시군구',
        'c09_사고유형_대분류',
        'c10_사고유형_중분류',
        'c11_사고유형',
        'c12_법규위반',
        'c13_도로형태_대분류',
        'c14_도로형태',
        'c15_당사자종별_1당_대분류',
        'c16_당사자종별_2당_대분류',
    ]

    # pprint(label_encoder_cols)
    # pprint(onehot_col)
    # pprint(x_cols)
    # pprint(origin_cols)

    def random_col_nullify(df, n_iter, ):
        rows = len(df)
        cols = len(df.columns)

        for _ in range(n_iter):
            while True:
                r = np.random.randint(0, rows)
                c = np.random.randint(0, cols)
                if df.loc[r, c] != 'none':
                    df.loc[r, c] = 'none'
                    break

    def duplicated_count(df):
        origin_df = DF(df[origin_cols])
        origin_df['dummy'] = np.zeros(shape=[len(origin_df)])
        pprint(origin_df.info())
        groupby = origin_df.groupby(origin_cols)['dummy'].count()
        # print(groupby)
        print(groupby.value_counts())


def col_mapper():
    path = "./data/samsung_contest/test_kor.csv"
    df = load_samsung(path)
    df = add_col_num(df, 2)
    cols = list(df.columns)
    alpha = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    to_col_name = dict(zip(alpha, cols))
    to_col_alpha = dict_transpose(to_col_name)

    return to_col_name, to_col_alpha


class samsung_full_set(BaseDataset):
    reg_cols = [
        'c02_사망자수',
        'c03_사상자수',
        'c04_중상자수',
        'c05_경상자수',
        'c06_부상신고자수',
    ]
    origin_cols = [
        'c00_주야',
        'c01_요일',
        'c02_사망자수',
        'c03_사상자수',
        'c04_중상자수',
        'c05_경상자수',
        'c06_부상신고자수',
        'c07_발생지시도',
        'c08_발생지시군구',
        'c09_사고유형_대분류',
        'c10_사고유형_중분류',
        'c11_사고유형',
        'c12_법규위반',
        'c13_도로형태_대분류',
        'c14_도로형태',
        'c15_당사자종별_1당_대분류',
        'c16_당사자종별_2당_대분류',
    ]

    def load(self, path):
        pickle_path = path_join(path, 'transformed.pkl')
        if not os.path.exists(pickle_path) or not self.caching:
            df = init_samsung(cache=True)

            self.transformer = transformer()
            df = self.transformer.transform(df)
            df = df.drop_duplicates(keep='first')

            for key in df:
                self.add_data(key, df[key])

            self.to_pickle(pickle_path)

            trans_csv_path = path_join(path, 'data_transformed.csv')
            save_samsung(df, trans_csv_path)
        else:
            self.from_pickle(pickle_path, overwrite_self=True)


def test_transform():
    dataset = samsung_contest(caching=False)
    dataset.load(path_head)
    trainset = dataset.pack['full_set']
    df = trainset.to_DataFrame()
    print(list(df.columns))
    x_keys = ['c00_주야', 'c01_요일', 'c02_사망자수', 'c03_사상자수', 'c04_중상자수', 'c05_경상자수', 'c06_부상신고자수', 'c07_발생지시도',
              'c08_발생지시군구',
              # 'c09_사고유형_대분류',
              'c10_사고유형_중분류', 'c11_법규위반', 'c13_도로형태', 'c12_도로형태_대분류', 'c14_당사자종별_1당_대분류',
              'c15_당사자종별_2당_대분류']
    y_keys = ['c09_사고유형_대분류']
    trainset.x_keys = x_keys
    trainset.y_keys = y_keys
    print(df.info())
    print(df.head())

    x, y = trainset.next_batch(10)
    print(x, y)
    print(x.shape, y.shape)

    x, y = trainset.next_batch(50, balanced_class=True)
    print(x, y)
    print(x.shape, y.shape)

    x, y = trainset.full_batch()
    print(trainset.classes)
    print(trainset.n_classes)
    print(trainset.size_group_by_class)
    print(x, y)
    print(x.shape, y.shape)
    print(trainset)

    a_set, b_set = trainset.split()
    print(a_set)
    print(b_set)


class samsung_contest(BaseDatasetPack):

    def __init__(self, caching=True, verbose=0, **kwargs):
        super().__init__(caching, verbose, **kwargs)
        self.pack['full_set'] = samsung_full_set(caching=caching, verbose=verbose, **kwargs)

    def load(self, path, limit=None, **kwargs):
        self.pack['full_set'].load(path, **kwargs)
