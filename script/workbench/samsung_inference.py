import math
import os
import numpy as np
from pprint import pprint
from script.data_handler.Base.BaseDataset import BaseDataset
from script.data_handler.samsun_contest import samsung_contest, path_head, load_samsung, add_col_num, save_samsung
from script.sklearn_like_toolkit.ClassifierPack import ClassifierPack
from script.util.PlotTools import PlotTools
from script.util.misc_util import path_join, load_pickle, dump_pickle
import pandas as pd

DF = pd.DataFrame


class SamsungInference:

    def __init__(self) -> None:
        super().__init__()

        self._p_types_str = None
        self._transformer = None
        self._test_df_trans = None
        self._dataset = None
        self._full_set = None
        self._test_df = None
        self._p_types = None
        self._model = None

        self.x_cols = list([
            # 'c00_주야',
            # 'c01_요일',
            'c02_사망자수',
            'c03_사상자수',
            'c04_중상자수',
            'c05_경상자수',
            'c06_부상신고자수',
            # 'c07_발생지시도',
            # 'c08_발생지시군구',
            'c09_사고유형_대분류',
            'c10_사고유형_중분류',
            'c11_법규위반',
            'c12_도로형태_대분류',
            'c13_도로형태',
            'c14_당사자종별_1당_대분류',
            'c15_당사자종별_2당_대분류',
        ])

    @property
    def dataset_pack(self):
        if self._dataset is None:
            self._dataset = samsung_contest(caching=False)
            self._dataset.load(path_head)

        return self._dataset

    @property
    def full_set(self):
        if self._full_set is None:
            self._full_set = self.dataset_pack['full_set']

        return self._full_set

    @property
    def test_df(self):
        if self._test_df is None:
            path = path_join(path_head, 'test_kor.csv')
            test_df = load_samsung(path)
            test_df = add_col_num(test_df)
            test_df = self.fill_rand(test_df)
            test_df = self.fill_inference_able(test_df)
            save_samsung(test_df, './test.csv')

            self._test_df = test_df

        return self._test_df

    @property
    def transformer(self):
        if self._transformer is None:
            self._transformer = self.full_set.transformer
        return self._transformer

    @property
    def test_df_trans(self):
        if self._test_df_trans is None:
            self._test_df_trans = self.transformer.transform(self.test_df)

        return self.test_df

    @property
    def p_types(self):
        if self._p_types is None:
            self._p_types = self.collect_p_types()

        return self._p_types

    @property
    def p_types_str(self):
        if self._p_types_str is None:
            self._p_types_str = [str(val) for val in self.p_types]

        return self._p_types_str

    @property
    def model(self):
        if self._model is None:
            self._model = self.train_models()

        return self._model

    def collect_p_types(self):
        test_df = self.test_df
        p_types = []
        for i in range(len(test_df)):
            p_types += [self.get_p_type(test_df, i)]
        p_types = sorted(p_types)
        p_types = list({str(value): value for value in p_types}.values())
        return p_types

    def get_p_type(self, df: DF, idx):
        df_row = df[idx:idx + 1]
        p_type = []
        for col in df.columns:
            if df_row[col].isna()[idx]:
                p_type += [col]
        return p_type

    def train_models(self, cache=True, path='./models.pkl'):
        p_types = self.p_types

        if os.path.exists(path) and cache:
            clfs = load_pickle(path)
            return clfs

        full_df = self.full_set.to_DataFrame()
        print(full_df.info())
        print('load data')

        p_types_str = [str(val) for val in p_types]
        pprint(p_types)

        clf_dict = {}
        for p_type, p_type_str in zip(p_types, p_types_str):
            print(f'train type : {p_type_str }')

            x_cols = list(self.x_cols)

            for y_col in p_type:
                x_cols.remove(y_col)

            clfs = {}
            for y_col in p_type:
                print(f'train label : {y_col}')
                print(x_cols, y_col)

                x_df = full_df[x_cols]
                y_df = full_df[[y_col]]
                dataset = BaseDataset(x=x_df, y=y_df)
                train_set, test_set = dataset.split()
                train_xs, train_ys = train_set.full_batch(out_type='df')
                test_xs, test_ys = test_set.full_batch(out_type='df')
                # print(train_xs.info())

                clf_name = 'skDecisionTreeClf'
                clf = ClassifierPack([clf_name])
                clf.fit(train_xs, train_ys)

                train_score = clf.score(train_xs, train_ys)
                test_score = clf.score(test_xs, test_ys)
                if len(train_score) == 0:
                    raise ValueError(f'{y_col} in {p_type} fail')
                pprint(f'score train = {train_score},\n test = {test_score}')

                predict = clf.predict(train_xs[:1])[clf_name]
                print(f'predict = {predict}, test_ys= {test_ys[:1]}')
                clfs[y_col] = clf

            clf_dict[p_type_str] = clfs

        dump_pickle(clf_dict, path)

        return clf_dict

    def predict(self, cache=True):
        test_df_trans = self.test_df_trans
        # p_types = self.p_types
        # p_types_str = self.p_types_str
        clf_dicts = self.model

        path = path_join(path_head, 'predict.csv')
        if os.path.exists(path) and cache:
            df = load_samsung(path)
            return df

        for idx in range(len(test_df_trans)):
            p_type = self.get_p_type(test_df_trans, idx)
            p_type_str = str(p_type)
            print(f'predict {idx}, {p_type_str}')

            x_cols = list(self.x_cols)

            for y_col in p_type:
                x_cols.remove(y_col)

            for y_col in p_type:
                print(f'predict {y_col}')

                x_df = DF(test_df_trans.loc[[idx], x_cols])
                x_df = x_df.reset_index(drop=True)

                clf = clf_dicts[p_type_str][y_col]
                print(clf)
                # print(x_df)
                predict = clf.predict(x_df)['skDecisionTreeClf'][0][0]
                test_df_trans.loc[idx, y_col] = predict
                print(f'predict = {predict}, at_df:{test_df_trans.loc[idx, y_col]}')

        save_samsung(test_df_trans, path)

        return test_df_trans

    def is_nan(self, val):
        try:
            if np.isnan(val):
                return True
            else:
                return False
        except BaseException as e:
            return False

    def fill_rand(self, test_df):
        path = path_join(path_head, 'data_init.csv')
        train_df = load_samsung(path)
        train_df['id'] = [0] * len(train_df)

        g = train_df.groupby(['c07_발생지시도', 'c08_발생지시군구'])['id'].count()
        mapper = {}
        for key, val in list(g.index):
            if key not in mapper:
                mapper[key] = [val]
            else:
                mapper[key] += [val]

        city_to_city_sub = mapper

        g = train_df.groupby('c07_발생지시도')['id'].count()
        city = list(g.index)

        g = train_df.groupby('c08_발생지시군구')['id'].count()
        city_sub = list(g.index)

        for idx in range(len(test_df)):
            if self.is_nan(test_df.loc[idx, 'c00_주야']):
                test_df.loc[idx, 'c00_주야'] = np.random.choice(['주간', '야간'], 1)[0]

            elif self.is_nan(test_df.loc[idx, 'c01_요일']):
                day = ['월', '화', '수', '목', '금', '토', '일']
                test_df.loc[idx, 'c01_요일'] = np.random.choice(day, 1)[0]

            elif self.is_nan(test_df.loc[idx, 'c07_발생지시도']) and self.is_nan(test_df.loc[idx, 'c08_발생지시군구']):
                test_df.loc[idx, 'c07_발생지시도'] = np.random.choice(city, 1)[0]
                test_df.loc[idx, 'c08_발생지시군구'] = np.random.choice(city_sub, 1)[0]

            elif not self.is_nan(test_df.loc[idx, 'c07_발생지시도']) and self.is_nan(test_df.loc[idx, 'c08_발생지시군구']):
                choice = city_to_city_sub[test_df.loc[idx, 'c07_발생지시도']]
                test_df.loc[idx, 'c08_발생지시군구'] = np.random.choice(choice, 1)[0]

        return test_df

    def fill_inference_able(self, test_df):
        for idx in range(len(test_df)):
            if self.is_nan(test_df.loc[idx, 'c09_사고유형_대분류']) and not self.is_nan(test_df.loc[idx, 'c15_당사자종별_2당_대분류']):
                mapper = {
                    '보행자': '차대사람',
                    '없음': '차량단독',
                }
                key = test_df.loc[idx, 'c15_당사자종별_2당_대분류']
                if key in mapper:
                    test_df.loc[idx, 'c09_사고유형_대분류'] = mapper[key]
                else:
                    test_df.loc[idx, 'c09_사고유형_대분류'] = '차대차'

            if self.is_nan(test_df.loc[idx, 'c12_도로형태_대분류']) and not self.is_nan(test_df.loc[idx, 'c13_도로형태']):
                if test_df.loc[idx, 'c13_도로형태'] == '기타단일로':
                    test_df.loc[idx, 'c12_도로형태_대분류'] = '단일로'
                else:
                    test_df.loc[idx, 'c12_도로형태_대분류'] = '교차로'

            if self.is_nan(test_df.loc[idx, 'c15_당사자종별_2당_대분류']) and not self.is_nan(test_df.loc[idx, 'c09_사고유형_대분류']):
                if test_df.loc[idx, 'c09_사고유형_대분류'] != '차대차' and test_df.loc[idx, 'c09_사고유형_대분류'] != '차대사람':
                    test_df.loc[idx, 'c15_당사자종별_2당_대분류'] = '없음'
                if test_df.loc[idx, 'c09_사고유형_대분류'] == '차대사람':
                    test_df.loc[idx, 'c15_당사자종별_2당_대분류'] = '보행자'

        return test_df

    def pipeline(self):
        test_df = self.test_df
        print(test_df.head())

        self.train_models(cache=False)

        predict_df = self.predict(cache=False)
        print(predict_df.head(5))

        predict_df = self.transformer.inverse_transform(predict_df)
        print(predict_df.head())
        save_samsung(predict_df, './predict.csv')

    def plot_all(self):
        path = "./data/samsung_contest/data_tansformed.csv"
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
            'c11_법규위반',
            'c12_도로형태_대분류',
            'c13_도로형태',
            'c14_당사자종별_1당_대분류',
            'c15_당사자종별_2당_대분류',
        ]

        # pprint(label_encoder_cols)
        # pprint(onehot_col)
        # pprint(x_cols)
        # pprint(origin_cols)

        plot = PlotTools()
        for key in origin_cols:
            # plot.dist(df, key, title=f'dist_{key}')
            plot.count(df, key, title=f'count_{key}')

        for a_key in origin_cols:
            for b_key in origin_cols:
                try:
                    plot.count(df, a_key, b_key, title=f'count_{a_key}_groupby_{b_key}')
                except BaseException as e:
                    print(a_key, b_key, e)

    def score_metric(self, df_true, df_predict):
        # TODO
        pass

    def categorical_score(self, y_true, y_predict, s=1):
        score = math.exp(- (((y_true - y_predict) / s) ** 2))
        return np.sum(score)
        pass

    def numerical_score(self, y_true, y_predict):
        return np.sum(y_true == y_predict)

    def hype_3(self):
        # TODO
        # one mlp to to all type
        pass

    def hype_4(self):
        # TODO
        # autoencoder or gan one model to all type
        pass
