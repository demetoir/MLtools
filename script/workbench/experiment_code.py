# -*- coding:utf-8 -*-

import os
import numpy as np
import pandas as pd
from tqdm import trange
from script.data_handler.DummyDataset import DummyDataset
from script.data_handler.titanic import build_transform
from script.sklearn_like_toolkit.ClassifierPack import ClassifierPack
from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.sklearn_like_toolkit.EnsembleClfPack import EnsembleClfPack
from script.sklearn_like_toolkit.FoldingHardVote import FoldingHardVote
from script.sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxStackingCVClf, mlxStackingClf
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from script.util.Logger import StdoutOnlyLogger, pprint_logger
from script.util.deco import deco_timeit, deco_save_log
from script.util.misc_util import path_join
from script.util.numpy_utils import np_stat_dict

########################################################################################################################
# print(built-in function) is not good for logging
from script.util.pandas_util import df_to_onehot_embedding, df_to_np_onehot_embedding

bprint = print
logger = StdoutOnlyLogger()
# noinspection PyShadowingBuiltins
print = logger.get_log()
pprint = pprint_logger(print)


#######################################################################################################################
@deco_timeit
@deco_save_log
def exp_stacking_metaclf(print, pprint):
    dataset = DatasetPackLoader().load_dataset("titanic")
    train_set, valid_set = dataset.split('train', 'train', 'valid', (7, 3))
    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])

    clf = ClassifierPack()
    clf.drop_clf('mlxMLP')
    clf.drop_clf('mlxAdaline')
    clf.drop_clf('mlxSoftmaxRegressionClf')
    clf.drop_clf('skGaussian_NB')
    clf.drop_clf('skQDA')

    pack = clf.pack
    for key, meta_clf in pack.items():
        if 'mlx' in key:
            continue

        pprint(f'meta clf = {key}')
        stacking = clf.make_stackingClf(meta_clf)
        stacking.fit(train_Xs, train_Ys)
        score = stacking.score(valid_Xs, valid_Ys)
        pprint(f'score {score}')
        # break


@deco_timeit
@deco_save_log
def exp_stackingCV_metaclf(print, pprint):
    dataset = DatasetPackLoader().load_dataset("titanic")
    train_set, valid_set = dataset.split('train', 'train', 'valid', (7, 3))
    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])

    clf = ClassifierPack()
    clf.drop_clf('mlxMLP')
    clf.drop_clf('mlxAdaline')
    clf.drop_clf('mlxSoftmaxRegressionClf')
    clf.drop_clf('skGaussian_NB')
    clf.drop_clf('skQDA')

    pack = clf.pack
    for key, meta_clf in pack.items():
        if 'mlx' in key:
            continue

        pprint(f'meta clf = {key}')
        stacking = clf.make_stackingCVClf(meta_clf)
        stacking.fit(train_Xs, train_Ys)
        score = stacking.score_pack(valid_Xs, valid_Ys)
        pprint(f'score {score}')
        # break


@deco_timeit
@deco_save_log
def exp_titanic_statistic(print, pprint):
    dataset = DatasetPackLoader().load_dataset("titanic")
    train_set, valid_set = dataset.split('train', 'train', 'valid', (7, 3))
    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])

    clf_pack = ClassifierPack()
    clf_pack.drop_clf('mlxMLP')
    clf_pack.drop_clf('mlxAdaline')
    clf_pack.drop_clf('mlxSoftmaxRegressionClf')
    clf_pack.drop_clf('skGaussian_NB')
    clf_pack.drop_clf('skQDA')

    pack = clf_pack.pack
    pprint(f'pack list {pack}')
    meta_clf = pack['skBernoulli_NB']
    pprint(f'metaclf = {meta_clf}')

    clf_pack.fit(train_Xs, train_Ys)
    score_pack = clf_pack.score_pack(valid_Xs, valid_Ys)
    pprint('default param clf pack')
    pprint(score_pack)

    clf_pack.param_search(train_Xs, train_Ys)
    score_pack = clf_pack.score_pack(valid_Xs, valid_Ys)
    pprint('optimize param clf pack top1')
    pprint(score_pack)

    pack = [clf for k, clf in clf_pack.pack.items() if hasattr(clf, 'get_params')]
    pack1_default = pack
    pack10_default = pack * 10
    pack100_default_ = pack * 100

    pack1_top1 = clf_pack.clone_top_k_tuned(k=1)
    pack1_top1 = [clf for k, clf in pack1_top1.items() if hasattr(clf, 'get_params')]
    pack10_top1 = pack1_top1 * 10
    pack100_top1 = pack1_top1 * 100

    pack1_top5 = clf_pack.clone_top_k_tuned(k=5)
    pack1_top5 = [clf for k, clf in pack1_top5.items() if hasattr(clf, 'get_params')]
    pack10_top5 = pack1_top5 * 10
    pack100_top5 = pack1_top5 * 100

    def voting_stacking_stackingCV(pack, param_type, pack_n, top):
        pprint(f'param_type={param_type}, pack_n={pack_n}, top={top}')

        voting = FoldingHardVote(pack)
        voting.fit(train_Xs, train_Ys)
        score_pack = voting.score_pack(valid_Xs, valid_Ys)
        pprint(f'{param_type} param clf pack * {pack_n}, {top} to hard voting')
        pprint(score_pack)

        stacking = mlxStackingClf(pack, meta_clf)
        stacking.fit(train_Xs, train_Ys)
        score_pack = stacking.score_pack(valid_Xs, valid_Ys)
        pprint(f'{param_type} param clf pack * {pack_n}, {top} to stacking')
        pprint(score_pack)

        stackingCV = mlxStackingCVClf(pack, meta_clf)
        stackingCV.fit(train_Xs, train_Xs)
        score_pack = stackingCV.score_pack(valid_Xs, valid_Ys)
        pprint(f'{param_type} param clf pack * {pack_n}, {top} to stackingCV')
        pprint(score_pack)

    voting_stacking_stackingCV(pack1_default, 'default', 1, None)
    voting_stacking_stackingCV(pack10_default, 'default', 10, None)
    voting_stacking_stackingCV(pack100_default_, 'default', 100, None)
    voting_stacking_stackingCV(pack1_top1, 'optimize', 1, 'top1')
    voting_stacking_stackingCV(pack10_top1, 'optimize', 10, 'top1')
    voting_stacking_stackingCV(pack100_top1, 'optimize', 100, 'top1')
    voting_stacking_stackingCV(pack1_top5, 'optimize', 1, 'top5')
    voting_stacking_stackingCV(pack10_top5, 'optimize', 10, 'top5')
    voting_stacking_stackingCV(pack100_top5, 'optimize', 100, 'top5')


@deco_timeit
@deco_save_log
def exp_titanic_id_static(print, pprint):
    dataset = DatasetPackLoader().load_dataset("titanic")
    dataset = dataset.set['train']

    ret_dict = {}
    n = 100
    for i in range(n):
        clf_pack = ClassifierPack()
        dataset.shuffle()
        train_set, valid_set = dataset.split((7, 3))
        train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
        clf_pack.fit(train_Xs, train_Ys)

        dataset.sort()
        full_Xs, full_Ys = dataset.full_batch(['Xs', 'Ys'])
        predict = clf_pack.predict(full_Xs)

        for key in predict:
            if key in ret_dict:
                ret_dict[key] += predict[key] / float(n)
            else:
                ret_dict[key] = predict[key] / float(n)

    import pandas as pd

    df = pd.DataFrame()
    for key in ret_dict:
        df[key] = ret_dict[key]
    for key in dataset.BATCH_KEYS:
        if key in ['Xs', 'Ys']:
            continue
        print(key, type(key))
        df[key] = dataset.full_batch([key])

    df.to_csv('./exp_titianic_id_result.csv', )


def print_score_statistic(clf, scores):
    stat = np_stat_dict(scores)
    pprint(f"""{clf}, {stat}""")


def exp_model_confidence():
    data_pack = DatasetPackLoader().load_dataset("titanic")
    # data_pack.shuffle()
    train_dataset = data_pack.set['train']
    train_dataset.sort()
    full_Xs, full_Ys = train_dataset.full_batch(['Xs', 'Ys'])

    train_set, valid_set = train_dataset.split((7, 3))
    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])

    dump_path = path_join('.', 'sklearn_pkl_dump')
    clf_pack_path = path_join(dump_path, ClassifierPack.__name__ + '.pkl')
    esm_pack_path = path_join(dump_path, EnsembleClfPack.__name__ + '.pkl')

    if not os.path.exists(clf_pack_path):
        clf_pack = ClassifierPack()
        clf_pack.param_search(train_Xs, train_Ys)
        clf_pack.dump(clf_pack_path)

    clf_pack = ClassifierPack().load(clf_pack_path)
    clf_pack.drop_clf('mlxAdaline')
    clf_pack.drop_clf('mlxLogisticRegression')
    clf_pack.drop_clf('skGaussian_NB')
    clf_pack.drop_clf('mlxMLP')
    clf_pack.drop_clf('skQDA')

    esm_pack = clf_pack.make_ensembleClfpack()
    esm_pack.fit(train_Xs, train_Ys)

    train_dataset.sort()
    full_Xs, full_Ys = train_dataset.full_batch(['Xs', 'Ys'])

    confidence_result_path = 'confidence_result.csv'
    df = pd.DataFrame()

    for key, val in esm_pack.predict_confidence(full_Xs).items():
        df[key] = val
    pprint(df.head(5))

    difficulty_path = './titanic_difficulty_stat.csv'
    difficulty_stat_df = pd.DataFrame().from_csv(difficulty_path)
    survived = difficulty_stat_df[['Survived']]
    difficulty_df = difficulty_stat_df[['total_difficulty']]
    df = pd.concat([df, difficulty_df, survived], axis=1)

    for key, predict in esm_pack.predict(full_Xs).items():
        if 'Vote' in key:
            df[key + '_score'] = np.equal(predict, survived['Survived']).astype(int)

    score = esm_pack.score_pack(full_Xs, full_Ys)
    pprint(score)

    # pprint(df.head(5))

    df.to_csv(confidence_result_path)


def exp_titanic_data_difficulty():
    data_pack = DatasetPackLoader().load_dataset("titanic")
    train_dataset = data_pack.set['train']
    train_dataset.shuffle()
    train_set, valid_set = train_dataset.split((7, 3))
    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])

    dump_path = path_join('.', 'sklearn_pkl_dump')
    clf_pack_path = path_join(dump_path, ClassifierPack.__name__ + '.pkl')
    if not os.path.exists(clf_pack_path):
        train_dataset.shuffle()
        train_set, valid_set = train_dataset.split((7, 3))
        train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])

        clf_pack = ClassifierPack()
        clf_pack.param_search(train_Xs, train_Ys)
        clf_pack.dump(clf_pack_path)

    clf_pack = ClassifierPack().load(clf_pack_path)
    clf_pack.drop_clf('mlxAdaline')
    clf_pack.drop_clf('mlxLogisticRegression')
    clf_pack.drop_clf('skGaussian_NB')
    clf_pack.drop_clf('mlxMLP')
    clf_pack.drop_clf('skQDA')

    def difficulty_stat(pack, dataset, n=100):
        dataset.sort()
        full_Xs, full_Ys = dataset.full_batch(['Xs', 'Ys'])

        ret = {}
        for clf_key, clf in pack.pack.items():
            print(clf_key)
            predicts = []
            for _ in trange(n):
                dataset.shuffle()
                train_set, valid_set = dataset.split((7, 3))
                train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])

                clf.fit(train_Xs, train_Ys)
                predict = clf.predict(full_Xs)
                predicts += [predict]

            predicts = np.array(predicts)
            predicts = predicts.transpose()

            stat_dicts = {'mean': [], 'std': []}
            for idx, predict in enumerate(predicts):
                row_stat = np_stat_dict(predict)
                for key in stat_dicts:
                    stat_dicts[key] += [row_stat[key]]

            for key in stat_dicts:
                ret[f'{clf_key}_{key}'] = stat_dicts[key]

        # pprint(ret.keys())
        # for key in ret:
        #     pprint(key, ret[key][:3], len(ret[key]))

        return ret

    result_path = './titanic_difficulty_stat.csv'
    if not os.path.exists(result_path):
        esm_pack = clf_pack.make_ensembleClfpack()
        esm_pack.fit(train_Xs, train_Ys)

        pack_dict = difficulty_stat(clf_pack, train_dataset, n=100)
        clf_pack_result_df = pd.DataFrame(pack_dict)
        esm_pack_dict = difficulty_stat(esm_pack, train_dataset, n=100)
        esm_pack_dict_df = pd.DataFrame(esm_pack_dict)

        train_dataset.sort()
        ground_true_df = train_dataset.to_DataFrame(['Survived'])

        result_df = pd.concat([ground_true_df, clf_pack_result_df, esm_pack_dict_df], axis=1)
        result_df.to_csv('./titanic_difficulty_stat.csv')

    df = pd.DataFrame.from_csv(result_path)

    # pprint(list(df.keys()))
    pprint(df.head(5))
    for key in df:
        if 'min' in key or 'max' in key:
            df = df.drop(columns=[key])
    df.to_csv(result_path)

    keys = [
        'skMLP_mean',
        'skMLP_std',
        'skSGD_mean',
        'skSGD_std',
        'skBernoulli_NB_mean',
        'skBernoulli_NB_std',
        'skMultinomial_NB_mean',
        'skMultinomial_NB_std',
        'skDecisionTree_mean',
        'skDecisionTree_std',
        'skRandomForest_mean',
        'skRandomForest_std',
        'skExtraTrees_mean',
        'skExtraTrees_std',
        'skAdaBoost_mean',
        'skAdaBoost_std',
        'skGradientBoosting_mean',
        'skGradientBoosting_std',
        'skKNeighbors_mean',
        'skKNeighbors_std',
        'skLinear_SVC_mean',
        'skLinear_SVC_std',
        'skRBF_SVM_mean',
        'skRBF_SVM_std',
        'skGaussianProcess_mean',
        'skGaussianProcess_std',
        'skBagging_mean',
        'skBagging_std',
        'XGBoost_mean',
        'XGBoost_std',
        'LightGBM_mean',
        'LightGBM_std',
        'CatBoost_mean',
        'CatBoost_std',
        'mlxPerceptronClf_mean',
        'mlxPerceptronClf_std',
        'FoldingHardVote_mean',
        'FoldingHardVote_std',
        'mlxStackingCVClf_mean',
        'mlxStackingCVClf_std',
        'mlxStackingClf_mean',
        'mlxStackingClf_std']
    for key in keys:
        if 'mean' in key:
            df[key] = abs(df['Survived'] - df[key])
    pprint(df.head(5))
    df.to_csv(result_path)

    df['total_difficulty'] = [0] * len(df)

    count = 0.0
    for key in keys:
        if 'mean' in key:
            count += 1
            df['total_difficulty'] += df[key]
    df['total_difficulty'] = df['total_difficulty'] / count
    pprint(df.head(5))
    df.to_csv(result_path)

    # col = clf_name_mean,clf_name_std ..., feature,
    # sort by difficulty
    # use best...


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


def exp_age_predict_regr():
    def trans_build_predict_age_dataset():
        path = os.getcwd()
        trans_merge_path = os.path.join(path, "data", "titanic", "trans_merge.csv")
        trans_df = pd.read_csv(trans_merge_path)

        merge_set_path = os.path.join(path, 'data', 'titanic', 'merge_set.csv')
        merge_set = pd.read_csv(merge_set_path)
        age = merge_set[['Age']]

        for col in trans_df.columns:
            if 'Unnamed' in col:
                del trans_df[col]
        # trans_df = trans_df.drop(columns=['Survived'])
        trans_df = trans_df.drop(columns=['Age_bucket'])

        trans_df = pd.concat([trans_df, age], axis=1)

        trans_df_with_age = trans_df.query("""not Age.isna()""")
        # pprint(trans_df_with_age.head(1))
        # pprint(trans_df_with_age.info())

        keys = list(trans_df_with_age.keys().values)

        keys.remove('Age')
        # pprint(keys)

        Xs_df = trans_df_with_age[keys]
        Ys_df = trans_df_with_age[['Age']]
        # pprint(list(Ys_df['Age'].unique()))
        # pprint(Xs_df.info())
        # pprint(Xs_df.head(5))
        #
        # pprint(Ys_df.info())
        # pprint(Ys_df.head(5))

        Xs = df_to_np_onehot_embedding(Xs_df)
        # Ys = df_to_np_onehot_embedding(Ys_df)
        Ys = np.array(Ys_df['Age'])
        # pprint(Xs.shape)
        # pprint(Ys.shape)
        dataset = DummyDataset()
        dataset.add_data('Xs', Xs)
        dataset.add_data('Ys', Ys)

        # pprint(dataset.to_DataFrame().info())

        return dataset

    def build_predict_age_dataset():
        path = os.getcwd()
        trans_merge_path = os.path.join(path, "data", "titanic", "trans_merge.csv")
        trans_df = pd.read_csv(trans_merge_path)

        for col in trans_df.columns:
            if 'Unnamed' in col:
                del trans_df[col]
        trans_df = trans_df.drop(columns=['Survived'])

        trans_df_with_age = trans_df.query("""Age_bucket != 'None' """)
        pprint(trans_df_with_age.head(1))
        pprint(trans_df_with_age.info())

        keys = list(trans_df_with_age.keys().values)

        keys.remove('Age_bucket')
        pprint(keys)

        Xs_df = trans_df_with_age[keys]
        Ys_df = trans_df_with_age[['Age_bucket']]
        pprint(list(Ys_df['Age_bucket'].unique()))
        # pprint(Xs_df.info())
        # pprint(Xs_df.head(5))

        pprint(Ys_df.info())
        pprint(Ys_df.head())

        Xs = df_to_np_onehot_embedding(Xs_df)
        Ys = df_to_np_onehot_embedding(Ys_df)
        # pprint(Xs.shape)
        # pprint(Ys.shape)
        dataset = DummyDataset()
        dataset.add_data('Xs', Xs)
        dataset.add_data('Ys', Ys)

        pprint(dataset.to_DataFrame().info())

        return dataset

    dataset = trans_build_predict_age_dataset()
    dataset.shuffle()
    train, test = dataset.split((20, 3))
    train_Xs, train_Ys = train.full_batch(['Xs', 'Ys'])
    test_Xs, test_Ys = test.full_batch(['Xs', 'Ys'])

    from xgboost import XGBRegressor
    reg = MLPRegressor(hidden_layer_sizes=(500,))
    reg = LGBMRegressor()
    reg = CatBoostRegressor()
    reg = XGBRegressor()
    reg.fit(train_Xs, train_Ys)
    pprint(reg.score(train_Xs, train_Ys))
    pprint(reg.score(test_Xs, test_Ys))
    n_sample = 5
    pprint(reg.predict(train_Xs[:n_sample]), train_Ys[:n_sample])
    pprint(reg.predict(test_Xs[:n_sample]), test_Ys[:n_sample])

    stat = abs(reg.predict(train_Xs) - train_Ys)
    stat = np_stat_dict(stat)
    pprint(stat)

    stat = abs(reg.predict(test_Xs) - test_Ys)
    stat = np_stat_dict(stat)
    pprint(stat)

    # clf_pack = ClassifierPack(['skMLP'])
    # clf_pack.pack['skMLP'].n_classes_ = test_Ys.shape[1]
    # # clf_pack.fit(train_Xs, train_Ys)
    # clf_pack.param_search(train_Xs, train_Ys)
    # pprint('train', clf_pack.score_pack(train_Xs, train_Ys))
    # pprint('test', clf_pack.score_pack(test_Xs, test_Ys))
    # pprint(['19~30',
    #         '30~40',
    #         '50~60',
    #         '1~3',
    #         '12~15',
    #         '3~6',
    #         '15~19',
    #         '6~12',
    #         '40~50',
    #         '60~81',
    #         '0~1'])


def main():
    print(exp_titanic_statistic.__name__)
    # exp_stacking_metaclf()
    # exp_voting()
    pass


def exp_titanic_corr_heatmap():
    import matplotlib.pyplot as plt
    import seaborn as sns

    def plot_corr_matrix(data, attr, fig_no):
        corr = data.corr()
        # f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.show()

    path = 'temp.csv'
    if not os.path.exists(path):
        df = build_transform(load_merge_set())
        df.to_csv(path, index=False)
        pprint(df.info())

    df = pd.read_csv(path, index_col=False)
    df = df.query('not Survived.isna()')
    pprint(df.info())

    df = df_to_onehot_embedding(df)
    pprint(df.info())

    corr = df.corr()
    # pprint(corr)
    pprint(list(corr['Survived_0.0'].items()))
    pprint(list(corr['Survived_1.0'].items()))
    # plot_corr_matrix(df, df.keys(), 3)


