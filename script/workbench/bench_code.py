# -*- coding:utf-8 -*-
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler

# print(built-in function) is not good for logging
from script.data_handler.MNIST import MNIST
from script.data_handler.wine_quality import wine_qualityPack
from script.model.sklearn_like_model.AE.AAE import AAE
from script.model.sklearn_like_model.AE.CVAE import CVAE
from script.model.sklearn_like_model.AE.VAE import VAE
from script.model.sklearn_like_model.GAN.C_GAN import C_GAN
from script.sklearn_like_toolkit.ClassifierPack import ClassifierPack
from script.util.Logger import pprint_logger, Logger
from script.util.PlotTools import PlotTools
from script.util.deco import deco_timeit
from script.util.misc_util import setup_file
from script.util.numpy_utils import np_img_to_tile, np_img_float32_to_uint8, np_image_save
from script.util.tensor_ops import activation_names
from unit_test.data_handler.test_wine_quality import load_wine_quality_dataset

bprint = print
logger = Logger('bench_code', level='INFO', )
print = logger.info
pprint = pprint_logger(print)
NpArr = np.array
DF = pd.DataFrame


# todo refactoring
def df_corr_to_value_tuple(corr):
    values = []

    keys = corr.keys()
    for i in range(len(keys)):
        for j in range(i):
            a, b = keys[i], keys[j]
            if a is not b:
                values += [(corr.loc[b, a], a, b)]

    return values


# todo refactoring
def test_timeit_np_arr_to_pd_df():
    size = 32 * 32
    np_arr = np.ones([32, 32])

    @deco_timeit
    def to_df(np_arr):
        df = pd.Series(np_arr)
        df = df.to_frame('np_arr')

        return df

    @deco_timeit
    def to_series(np_arr):
        df = pd.Series(np_arr)
        return df

    df = to_df(np_arr)
    series = to_series(np_arr)
    print(series)


# todo refactoring
def params_to_dict(**param):
    return param


# todo refactoring
def to_list_grid(grid, depth=0, recursive=True):
    for key, val in grid.items():
        if type(val) is dict and recursive:
            grid[key] = to_list_grid(val, depth=depth + 1)

    if depth is not 0:
        grid = ParameterGrid(grid)
    return grid


# todo refactoring
def param_grid_random(grid, n_iter):
    return ParameterSampler(to_list_grid(grid), n_iter=n_iter)


# todo refactoring
def param_grid_full(grid):
    return ParameterGrid(to_list_grid(grid))


# todo refactoring
def test_param_grid_random():
    BOOLs = [True, False]
    encoder_kwargs = params_to_dict(
        linear_stack_bn=BOOLs, linear_stack_activation=activation_names,
        tail_bn=BOOLs, tail_activation=activation_names,
    )
    decoder_kwargs = params_to_dict(
        linear_stack_bn=BOOLs, linear_stack_activation=activation_names,
        tail_bn=BOOLs, tail_activation=activation_names,
    )

    param_grid = params_to_dict(
        latent_code_size=[2],
        learning_rate=[0.01],
        encoder_net_shapes=[(512, 256, 128, 64, 32)],
        decoder_net_shapes=[(32, 64, 128, 256, 512)],
        batch_size=[256],
        KL_D_rate=[0.01],
        # loss_type=VAE.loss_names,
        loss_type=['VAE'],
        encoder_kwargs=encoder_kwargs,
        # decoder_kwargs=decoder_kwargs
    )

    param_grid = param_grid_random(param_grid, 50)

    for grid in param_grid:
        pprint(grid)
    pprint(len(param_grid))


# todo refactoring
def test_param_grid_full():
    BOOLs = [True, False]
    encoder_kwargs = params_to_dict(
        linear_stack_bn=BOOLs, linear_stack_activation=activation_names,
        tail_bn=BOOLs, tail_activation=activation_names,
    )
    decoder_kwargs = params_to_dict(
        linear_stack_bn=BOOLs, linear_stack_activation=activation_names,
        tail_bn=BOOLs, tail_activation=activation_names,
    )

    param_grid = params_to_dict(
        latent_code_size=[2],
        learning_rate=[0.01],
        encoder_net_shapes=[(512, 256, 128, 64, 32)],
        decoder_net_shapes=[(32, 64, 128, 256, 512)],
        batch_size=[256],
        KL_D_rate=[0.01],
        loss_type=['VAE'],
        encoder_kwargs=encoder_kwargs,
        # decoder_kwargs=decoder_kwargs
    )

    param_grid = param_grid_full(param_grid)

    for grid in param_grid:
        pprint(grid)
    pprint(len(param_grid))


def test_wine_quality_predict():
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


def test_data_VAE_aug_wine_quality():
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


def test_data_GAN_aug_wine_quality():
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


def test_wine_quality_pca():
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


def test_VAE_latent_space():
    dataset_path = """C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\data\\MNIST"""
    dataset_pack = MNIST().load(dataset_path)
    dataset_pack.shuffle()
    train_set = dataset_pack['train']
    full_Xs, full_Ys = train_set.full_batch()

    x = train_set.Ys_index_label
    idxs_labels = []
    for i in range(10):
        idxs_label = np.where(x == i)
        idxs_labels += [(full_Xs[idxs_label], full_Ys[idxs_label])]

    plot = PlotTools()

    # model = CVAE
    model = VAE
    save_path = './CVAE.pkl'

    n_iter = 6

    BOOLs = [True, False]
    linear_stack_activation = ['sigmoid', 'tanh', 'relu', 'lrelu', 'elu']
    param_grid = params_to_dict(

        latent_code_size=[2],
        learning_rate=[0.005],
        encoder_net_shapes=[(512, 256, 128, 64, 32)],
        decoder_net_shapes=[(32, 64, 128, 256, 512)],
        batch_size=[256],
        # KL_D_rate=[0.01],
        loss_type=['VAE'],
        encoder_kwargs=params_to_dict(
            linear_stack_bn=[False],
            linear_stack_activation=['relu', 'lrelu', 'elu'],
            tail_bn=[False],
            tail_activation=['lrelu', 'none'],
        ),
        decoder_kwargs=params_to_dict(
            linear_stack_bn=[True],
            linear_stack_activation=['relu'],
            tail_bn=[False],
            tail_activation=['sigmoid'],
        )
    )
    param_grid = param_grid_full(param_grid)

    df = DF({
        'params': list(param_grid)
    })
    df.to_csv('./params.csv')

    for param_idx, params in enumerate(param_grid):
        pprint(param_idx, params)

        ae = model(**params)
        for i in range(50):
            # ae.train(full_Xs, full_Ys, epoch=1)
            ae.train(full_Xs, epoch=1)
            # metric = ae.metric(full_Xs, full_Ys)
            # ae.save(save_path)
            metric = ae.metric(full_Xs)
            if np.isnan(metric):
                print(f'param_idx:{param_idx}, metric is {metric}')
                break
            print(metric)

            codes = []
            for x, y in idxs_labels:
                # code = ae.code(x, y)
                code = ae.code(x)
                codes += [code]

            plot.scatter_2d(*codes, title=f'param_idx_{param_idx}_vae_latent_space_epoch_{i}.png')

        del ae


def test_VAE():
    dataset_path = """C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\data\\MNIST"""
    dataset_pack = MNIST().load(dataset_path)
    dataset_pack.shuffle()
    train_set = dataset_pack['train']
    full_Xs, full_Ys = train_set.full_batch()

    x = train_set.Ys_index_label
    idxs_labels = []
    for i in range(10):
        idxs_label = np.where(x == i)
        idxs_labels += [(full_Xs[idxs_label], full_Ys[idxs_label])]

    plot = PlotTools()

    model = VAE
    params = {
        'loss_type':          'VAE',
        'learning_rate':      0.01,
        'latent_code_size':   2,
        'encoder_net_shapes': (512, 256, 128, 64, 32),
        'encoder_kwargs':     {
            'tail_bn':                 False,
            'tail_activation':         'lrelu',
            'linear_stack_bn':         False,
            'linear_stack_activation': 'lrelu',
        },
        'decoder_net_shapes': (32, 64, 128, 256, 512),
        'decoder_kwargs':     {
            'tail_bn':                 True,
            'tail_activation':         'sigmoid',
            'linear_stack_bn':         True,
            'linear_stack_activation': 'relu'
        },
        'batch_size':         256,
        # 'KL_D_rate': 0.01
    }
    # params = {'loss_type': 'VAE', 'learning_rate': 0.01, 'latent_code_size': 2,
    #           'encoder_net_shapes': (512, 256, 128, 64, 32),
    #           'encoder_kwargs': {'tail_bn': True, 'tail_activation': 'elu', 'linear_stack_bn': True,
    #                              'linear_stack_activation': 'lrelu'},
    #           'decoder_net_shapes': (32, 64, 128, 256, 512),
    #           'decoder_kwargs': {'tail_bn': True, 'tail_activation': 'relu', 'linear_stack_bn': True,
    #                              'linear_stack_activation': 'tanh'}, 'batch_size': 256, 'KL_D_rate': 0.01
    #           }
    ae = model(**params)
    n_iter = 100
    for i in range(n_iter):
        # ae.train(full_Xs, full_Ys, epoch=1)
        ae.train(full_Xs, epoch=1)
        # metric = ae.metric(full_Xs, full_Ys)
        # ae.save(save_path)
        metric = ae.metric(full_Xs)
        if np.isnan(metric):
            print(f'metric is {metric}')
            break
        print(metric)

        codes = []
        for x, y in idxs_labels:
            # code = ae.code(x, y)
            code = ae.code(x)
            codes += [code]

        plot.scatter_2d(*codes, title=f'vae_latent_space_epoch_{i}.png')

    del ae


def test_CVAE():
    dataset_path = """C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\data\\MNIST"""
    dataset_pack = MNIST().load(dataset_path)
    dataset_pack.shuffle()
    train_set = dataset_pack['train']
    full_Xs, full_Ys = train_set.full_batch()

    x = train_set.Ys_index_label
    idxs_labels = []
    for i in range(10):
        idxs_label = np.where(x == i)
        idxs_labels += [(full_Xs[idxs_label], full_Ys[idxs_label])]

    plot = PlotTools()

    model = CVAE
    params = {
        'loss_type':          'VAE',
        'learning_rate':      0.01,
        'latent_code_size':   2,
        'encoder_net_shapes': (512, 256, 128, 64, 32),
        'encoder_kwargs':     {
            'tail_bn':                 False,
            'tail_activation':         'none',
            'linear_stack_bn':         False,
            'linear_stack_activation': 'lrelu',
        },
        'decoder_net_shapes': (32, 64, 128, 256, 512),
        'decoder_kwargs':     {
            'tail_bn':                 True,
            'tail_activation':         'sigmoid',
            'linear_stack_bn':         True,
            'linear_stack_activation': 'relu'
        },
        'batch_size':         256,
        # 'KL_D_rate': 0.01
    }
    # params = {'loss_type': 'VAE', 'learning_rate': 0.01, 'latent_code_size': 2,
    #           'encoder_net_shapes': (512, 256, 128, 64, 32),
    #           'encoder_kwargs': {'tail_bn': True, 'tail_activation': 'elu', 'linear_stack_bn': True,
    #                              'linear_stack_activation': 'lrelu'},
    #           'decoder_net_shapes': (32, 64, 128, 256, 512),
    #           'decoder_kwargs': {'tail_bn': True, 'tail_activation': 'relu', 'linear_stack_bn': True,
    #                              'linear_stack_activation': 'tanh'}, 'batch_size': 256, 'KL_D_rate': 0.01
    #           }
    ae = model(**params)
    n_iter = 50
    for i in range(n_iter):
        ae.train(full_Xs, full_Ys, epoch=1)
        metric = ae.metric(full_Xs, full_Ys)
        # ae.save(save_path)
        if np.isnan(metric):
            print(f'metric is {metric}')
            break
        print(metric)

        codes = []
        for x, y in idxs_labels:
            code = ae.code(x, y)
            codes += [code]

        plot.scatter_2d(*codes, title=f'CVAE_latent_space_epoch_{i}.png')
        for label, code in enumerate(codes):
            plot.scatter_2d(code, title=f'vae_latent_space_epoch_{i}_label+{label}.png')

    del ae


def test_CVAE_latent_space():
    dataset_path = """C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\data\\MNIST"""
    dataset_pack = MNIST().load(dataset_path)
    dataset_pack.shuffle()
    train_set = dataset_pack['train']
    full_Xs, full_Ys = train_set.full_batch()
    sample_Xs, sample_Ys = full_Xs[:5], full_Ys[:5]

    x = train_set.Ys_index_label
    idxs_labels = []
    for i in range(10):
        idxs_label = np.where(x == i)
        idxs_labels += [(full_Xs[idxs_label], full_Ys[idxs_label])]

    plot = PlotTools()

    model = CVAE
    save_path = './CVAE.pkl'

    BOOLs = [True, False]
    linear_stack_activation = ['sigmoid', 'tanh', 'relu', 'lrelu', 'elu']
    param_grid = params_to_dict(

        latent_code_size=[2],
        learning_rate=[0.005],
        encoder_net_shapes=[(512, 256, 128, 64, 32)],
        decoder_net_shapes=[(32, 64, 128, 256, 512)],
        batch_size=[256],
        # KL_D_rate=[0.01],
        loss_type=['VAE'],
        encoder_kwargs=params_to_dict(
            linear_stack_bn=[False],
            linear_stack_activation=['relu', 'lrelu', 'elu'],
            tail_bn=[False],
            tail_activation=['lrelu', 'none'],
        ),
        decoder_kwargs=params_to_dict(
            linear_stack_bn=[True],
            linear_stack_activation=['relu'],
            tail_bn=[False],
            tail_activation=['sigmoid'],
        )
    )
    param_grid = param_grid_full(param_grid)

    df = DF({
        'params': list(param_grid)
    })
    df.to_csv('./params.csv')

    for param_idx, params in enumerate(param_grid):
        pprint(param_idx, params)

        ae = model(**params)
        for i in range(10):
            ae.train(full_Xs, full_Ys, epoch=1)
            # ae.save(save_path)
            metric = ae.metric(full_Xs, full_Ys)
            if np.isnan(metric):
                print(f'param_idx:{param_idx}, metric is {metric}')
                break
            print(metric)

            codes = []
            for x, y in idxs_labels:
                code = ae.code(x, y)
                codes += [code]

            plot.scatter_2d(*codes, title=f'param_idx_{param_idx}/vae_latent_space_epoch_{i}.png')
            # for label, code in enumerate(codes):
            #     plot.scatter_2d(code, title=f'param_idx_{param_idx}/vae_latent_space_epoch_{i}_label+{label}.png')

            recon = ae.recon(sample_Xs, sample_Ys)
            gen = ae.generate(sample_Ys)
            np_img = np.concatenate([sample_Xs, recon, gen])
            np_img = np_img_float32_to_uint8(np_img)

            # sample_imgs = Xs_gen
            file_name = f'./matplot/param_idx_{param_idx}/vae_img_epoch_{i}.png'
            tile = np_img_to_tile(np_img, column_size=5)
            np_image_save(tile, file_name)

        del ae


def test_AAE_latent_space():
    dataset_path = """C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\data\\MNIST"""
    dataset_pack = MNIST().load(dataset_path)
    dataset_pack.shuffle()
    train_set = dataset_pack['train']
    full_Xs, full_Ys = train_set.full_batch()
    sample_Xs, sample_Ys = full_Xs[:5], full_Ys[:5]

    x = train_set.Ys_index_label
    idxs_labels = []
    for i in range(10):
        idxs_label = np.where(x == i)
        idxs_labels += [(full_Xs[idxs_label], full_Ys[idxs_label])]

    plot = PlotTools()

    model = AAE
    save_path = './CVAE.pkl'

    BOOLs = [True, False]
    linear_stack_activation = ['sigmoid', 'tanh', 'relu', 'lrelu', 'elu']
    param_grid = params_to_dict(

        latent_code_size=[2],
        learning_rate=[0.01],
        encoder_net_shapes=[(512, 256)],
        decoder_net_shapes=[(256, 512)],
        batch_size=[100],
        # KL_D_rate=[0.01],
        encoder_kwargs=params_to_dict(
            linear_stack_bn=[False],
            linear_stack_activation=['relu'],
            tail_bn=[False],
            tail_activation=['none'],
        ),
        decoder_kwargs=params_to_dict(
            linear_stack_bn=[False],
            linear_stack_activation=['relu'],
            tail_bn=[False],
            tail_activation=['sigmoid'],
        )
    )
    param_grid = param_grid_full(param_grid)

    df = DF({
        'params': list(param_grid)
    })
    df.to_csv('./params.csv')

    for param_idx, params in enumerate(param_grid):
        pprint(param_idx, params)

        ae = model(**params)
        for i in range(100):
            ae.train(full_Xs, full_Ys, epoch=1)
            # ae.save(save_path)
            metric = ae.metric(full_Xs, full_Ys)
            if any([np.isnan(val) for val in metric.values()]):
                print(f'param_idx:{param_idx}, metric is {metric}')
                break
            print(metric)

            codes = []
            for x, y in idxs_labels:
                code = ae.code(x)
                codes += [code]

            plot.scatter_2d(*codes, title=f'param_idx_{param_idx}/aae_latent_space_epoch_{i}.png')
            # for label, code in enumerate(codes):
            #     plot.scatter_2d(code, title=f'param_idx_{param_idx}/aae_latent_space_epoch_{i}_label+{label}.png')

            recon = ae.recon(sample_Xs, sample_Ys)
            gen = ae.generate(full_Ys[:30])
            code_walk = np.concatenate([ae.augmentation(sample_Xs, sample_Ys) for _ in range(5)], axis=0)
            recon_sharpen = ae.recon_sharpen(sample_Xs, sample_Ys)
            np_img = np.concatenate([sample_Xs, recon, recon_sharpen, gen, code_walk])

            def plot_image(np_img, path):

                setup_file(path)
                np_image_save(np_img, path)

            def plot_image_tile(np_imgs, path, column=10):
                setup_file(path)
                np_img_tile = np_img_to_tile(np_imgs, column_size=column)
                np_image_save(np_img_tile, path)

                pass

            np_img = np_img_float32_to_uint8(np_img)
            file_name = f'./matplot/param_idx_{param_idx}/aae_img_epoch_{i}.png'
            plot_image_tile(np_img, file_name, column=5)
            # sample_imgs = Xs_gen

        del ae


@deco_timeit
def main():
    test_AAE_latent_space()
    # test_CVAE_latent_space()

    pass
