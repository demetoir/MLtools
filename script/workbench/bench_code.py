# -*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from script.data_handler.wine_quality import wine_qualityPack
from script.model.sklearn_like_model.AE.CVAE import CVAE
from script.model.sklearn_like_model.GAN.C_GAN import C_GAN
from script.sklearn_like_toolkit.ClassifierPack import ClassifierPack
from script.util.Logger import pprint_logger, Logger
from script.util.PlotTools import PlotTools
from script.util.deco import deco_timeit
from unit_test.data_handler.test_wine_quality import load_wine_quality_dataset

# print(built-in function) is not good for logging
bprint = print
logger = Logger('bench_code', level='INFO', )
print = logger.info
pprint = pprint_logger(print)
NpArr = np.array
DF = pd.DataFrame


def exp_wine_quality_predict():
    dataset_path = """C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\data\\winequality"""
    dataset_pack = wine_qualityPack().load(dataset_path)
    dataset_pack.shuffle()
    dataset_pack.split('data', 'train', 'test', rate=(7, 3))
    train_set = dataset_pack['train']
    test_set = dataset_pack['test']
    print(train_set.to_DataFrame().info())

    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    test_Xs, test_Ys = test_set.full_batch(['Xs', 'Ys'])

    import os

    clf_pack_path = './clf_pack.pkl'
    if not os.path.exists(clf_pack_path):
        clf_pack = ClassifierPack(['XGBoostClf'])
        # clf_pack.drop('CatBoostClf')
        # clf_pack.drop('skQDAClf')
        # clf_pack.drop('skNearestCentroidClf')
        # clf_pack.drop('skGaussianProcessClf')
        # clf_pack.drop('skGaussian_NBClf')
        # clf_pack.drop('skGradientBoostingClf')
        # clf_pack.drop('skMultinomial_NBClf')
        # clf_pack.drop('skPassiveAggressiveClf')

        # clf_pack.fit(train_Xs, train_Ys)
        clf_pack.HyperOptSearch(train_Xs, train_Ys, 200, parallel=False)

        clf_pack.dump(clf_pack_path)
    else:
        clf_pack = ClassifierPack().load(clf_pack_path)

    clf_pack = ClassifierPack(['XGBoostClf'])
    clf_pack.fit(train_Xs, train_Ys)

    score = clf_pack.score(train_Xs, train_Ys)
    pprint(score)

    score = clf_pack.score(test_Xs, test_Ys)
    pprint(score)

    # proba = clf_pack.predict_proba(train_Xs[:5])
    # print(proba)

    smaple_X = test_Xs[10:15]
    sample_y = test_Ys[10:15]

    proba = clf_pack.predict_proba(smaple_X)
    print(proba)
    predict = clf_pack.predict(smaple_X)
    print(predict)
    print(sample_y)

    # score = clf_pack.score_pack(train_Xs, train_Ys)
    # pprint(score)
    #
    score = clf_pack.score_pack(test_Xs, test_Ys)
    pprint(score)
    matrix = score['XGBoostClf']['confusion_matrix']
    pprint(matrix / len(test_Xs))


def exp_data_aug_VAE_wine_quality():
    rets = load_wine_quality_dataset()
    Xs, Ys, test_Xs, test_Ys, sample_xs, sample_Ys = rets

    CVAE_path = './CVAE.pkl'
    if not os.path.exists(CVAE_path):
        cvae = CVAE(loss_type='MSE_only', learning_rate=0.01, latent_code_size=5,
                    decoder_net_shapes=(128, 128), encoder_net_shapes=(128, 128), batch_size=512)
        cvae.train(Xs, Ys, epoch=1600)
        cvae.save(CVAE_path)
    else:
        cvae = CVAE().load(CVAE_path)

    metric = cvae.metric(Xs, Ys)
    metric = np.mean(metric)
    print(metric)

    def train_doubling_batch_size(tf_model, Xs, Ys, epoch=100, iter=3):
        for _ in range(iter):
            tf_model.train(Xs, Ys, epoch=epoch)
            metric = tf_model.metric(Xs, Ys)
            metric = np.mean(metric)
            print(metric)
            tf_model.batch_size *= 2
            epoch *= 2

    recon = cvae.recon(sample_xs, sample_Ys)
    print(sample_xs)
    pprint(recon)

    gen_labels_1 = np.array([[0, 1] for _ in range(2000)])
    gen_x_1 = cvae.generate(gen_labels_1)
    merged_Xs = np.concatenate([Xs, gen_x_1])
    merged_Ys = np.concatenate([Ys, gen_labels_1])

    gen_labels_0 = np.array([[1, 0] for _ in range(2000)])
    gen_x_0 = cvae.generate(gen_labels_1)
    gen_only_x = np.concatenate([gen_x_0, gen_x_1])
    gen_only_labels = np.concatenate([gen_labels_0, gen_labels_1])

    high = 0.02
    low = -0.02
    noised_Xs = np.concatenate([
        Xs,
        Xs + np.random.uniform(low, high, size=Xs.shape),
        Xs + np.random.uniform(low, high, size=Xs.shape),
        Xs + np.random.uniform(low, high, size=Xs.shape),
    ])
    noised_Ys = np.concatenate([Ys, Ys, Ys, Ys])

    def print_score_info(clf_pack):
        merge_score = clf_pack.score(gen_only_x, gen_only_labels)
        train_score = clf_pack.score(Xs, Ys)
        test_score = clf_pack.score(test_Xs, test_Ys)
        noised_score = clf_pack.score(noised_Xs, noised_Ys)
        print('aug score')
        pprint(f' train_score : {train_score}')
        pprint(f'merge_score : {merge_score}')
        pprint(f'test_score : {test_score}')
        pprint(f'noised_score : {noised_score}')

    clf_pack = ClassifierPack(['XGBoostClf'])
    clf_pack.fit(merged_Xs, merged_Ys)
    print('fit merge')
    print_score_info(clf_pack)

    clf_pack = ClassifierPack(['XGBoostClf'])
    clf_pack.fit(Xs, Ys)
    print('fit origin')
    print_score_info(clf_pack)

    clf_pack = ClassifierPack(['XGBoostClf'])
    clf_pack.fit(gen_only_x, gen_only_labels)
    print('fit gen_only')
    print_score_info(clf_pack)

    clf_pack = ClassifierPack(['XGBoostClf'])
    clf_pack.fit(noised_Xs, noised_Ys)
    print('fit noised')
    print_score_info(clf_pack)


def exp_data_aug_GAN_wine_quality():
    rets = load_wine_quality_dataset()
    train_Xs, train_Ys, test_Xs, test_Ys, sample_xs, sample_Ys = rets

    model = C_GAN
    GAN_path = f'./{model.__name__}.pkl'
    if not os.path.exists(GAN_path):
        c_GAN = model(loss_type='WGAN', learning_rate=0.001, batch_size=256, G_net_shape=(256, 256),
                      D_net_shape=(64, 64))
        c_GAN.train(train_Xs, train_Ys, epoch=200)
        c_GAN.save(GAN_path)
    else:
        c_GAN = model().load(GAN_path)

    metric = c_GAN.metric(train_Xs, train_Ys)
    print(metric)

    recon = c_GAN.recon(sample_xs, sample_Ys)
    print(sample_xs)
    pprint(recon)

    gen_labels_1 = np.array([[0, 1] for _ in range(2000)])
    gen_x_1 = c_GAN.generate(gen_labels_1)
    merged_Xs = np.concatenate([train_Xs, gen_x_1])
    merged_Ys = np.concatenate([train_Ys, gen_labels_1])

    gen_labels_0 = np.array([[1, 0] for _ in range(2000)])
    gen_x_0 = c_GAN.generate(gen_labels_1)
    gen_only_x = np.concatenate([gen_x_0, gen_x_1])
    gen_only_labels = np.concatenate([gen_labels_0, gen_labels_1])

    high = 0.02
    low = -0.02
    noised_Xs = np.concatenate([
        train_Xs,
        train_Xs + np.random.uniform(low, high, size=train_Xs.shape),
        train_Xs + np.random.uniform(low, high, size=train_Xs.shape),
        train_Xs + np.random.uniform(low, high, size=train_Xs.shape),
    ])
    noised_Ys = np.concatenate([train_Ys, train_Ys, train_Ys, train_Ys])

    def print_score_info(clf_pack):
        merge_score = clf_pack.score(gen_only_x, gen_only_labels)
        train_score = clf_pack.score(train_Xs, train_Ys)
        test_score = clf_pack.score(test_Xs, test_Ys)
        noised_score = clf_pack.score(noised_Xs, noised_Ys)
        print('aug score')
        pprint(f' train_score : {train_score}')
        pprint(f'merge_score : {merge_score}')
        pprint(f'test_score : {test_score}')
        pprint(f'noised_score : {noised_score}')

    clf_pack = ClassifierPack(['XGBoostClf'])
    clf_pack.fit(merged_Xs, merged_Ys)
    print('fit merge')
    print_score_info(clf_pack)

    clf_pack = ClassifierPack(['XGBoostClf'])
    clf_pack.fit(train_Xs, train_Ys)
    print('fit origin')
    print_score_info(clf_pack)

    clf_pack = ClassifierPack(['XGBoostClf'])
    clf_pack.fit(gen_only_x, gen_only_labels)
    print('fit gen_only')
    print_score_info(clf_pack)

    clf_pack = ClassifierPack(['XGBoostClf'])
    clf_pack.fit(noised_Xs, noised_Ys)
    print('fit noised')
    print_score_info(clf_pack)


def groupby_label(dataset):
    # x = dataset_pack['data'].Ys_onehot_label
    x = dataset.Ys_index_label

    print(np.bincount(x))
    print(np.unique(x))

    label_partials = []
    for key in np.unique(x):
        idxs = np.where(x == key)
        partial = dataset.Xs[idxs]
        label_partials += [partial]

    return label_partials


def exp_wine_quality_pca():
    df_Xs_keys = [
        'col_0_fixed_acidity', 'col_1_volatile_acidity', 'col_2_citric_acid',
        'col_3_residual_sugar', 'col_4_chlorides', 'col_5_free_sulfur_dioxide',
        'col_6_total_sulfur_dioxide', 'col_7_density', 'col_8_pH',
        'col_9_sulphates', 'col_10_alcohol', 'col_12_color'
    ]
    df_Ys_key = 'col_11_quality'

    dataset_path = """C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\data\\winequality"""
    dataset_pack = wine_qualityPack().load(dataset_path)
    dataset_pack.shuffle()
    full_Xs, full_Ys = dataset_pack['data'].full_batch(['Xs', 'Ys'])

    dataset_pack.split('data', 'train', 'test', rate=(7, 3))
    train_set = dataset_pack['train']
    test_set = dataset_pack['test']

    Xs, Ys = train_set.full_batch(['Xs', 'Ys'])
    test_Xs, test_Ys = test_set.full_batch(['Xs', 'Ys'])
    sample_xs = Xs[:5]
    sample_Ys = Ys[:5]

    label_partials = groupby_label(dataset_pack['data'])

    model = PCA(n_components=2)
    transformed = model.fit(full_Xs)
    print(transformed)

    transformed = []
    for partial in label_partials:
        pca_partial = model.transform(partial)
        transformed += [pca_partial]

    pprint(transformed)
    plot = PlotTools(show=True, save=False)
    plot.scatter_2d(*transformed)


@deco_timeit
def main():
    pass
