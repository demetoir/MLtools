# -*- coding:utf-8 -*-
# from script.model.sklearn_like_model.AE.CVAE import CVAE
import os
import matplotlib.pyplot as plt
import seaborn as sns



from script.model.sklearn_like_model.GAN.C_GAN import C_GAN
from script.model.sklearn_like_model.AE.AAE import AAE
from script.model.sklearn_like_model.AE.CVAE import CVAE
from script.model.sklearn_like_model.AE.VAE import VAE
from script.model.sklearn_like_model.MLPClassifier import MLPClassifier
from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.model.sklearn_like_model.AE.AE import AE
from script.sklearn_like_toolkit.ClassifierPack import ClassifierPack
from script.model.sklearn_like_model.GAN.GAN import GAN
from script.util.Logger import pprint_logger, Logger
from script.util.deco import deco_timeit
import numpy as np

########################################################################################################################
# print(built-in function) is not good for logging
from unit_test.model.sklearn_like_model.AE.test_VAE import test_VAE
from script.util.misc_util import path_join, time_stamp, setup_file

bprint = print
logger = Logger('bench_code', level='INFO', )
print = logger.info
pprint = pprint_logger(print)


#######################################################################################################################


def onehot_label_smooth(Ys, smooth=0.05):
    Ys *= (1 - smooth * 2)
    Ys += smooth
    return Ys


def finger_print(size, head='_'):
    ret = head
    h_list = [c for c in '01234556789qwertyuiopasdfghjklzxcvbnm']
    for i in range(size):
        ret += np.random.choice(h_list)
    return ret


def titanic_submit():
    datapack = DatasetPackLoader().load_dataset('titanic')
    # datapack = DatasetPackLoader().load_dataset('titanic')
    train_set = datapack['train']
    test_set = datapack['test']
    train_set.shuffle()
    train, valid = train_set.split()

    train_Xs, train_Ys = train.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid.full_batch(['Xs', 'Ys'])

    path = './clf_pack.clf'
    if not os.path.exists(path):
        train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
        clf_pack = ClassifierPack()
        clf_pack.gridSearchCV(train_Xs, train_Ys, cv=10)
        clf_pack.dump(path)

    clf_pack = ClassifierPack().load(path)
    # pprint(clf_pack.optimize_result)
    clf_pack.drop_clf('skQDA')
    clf_pack.drop_clf('skGaussian_NB')
    clf_pack.drop_clf('mlxSoftmaxRegressionClf')
    clf_pack.drop_clf('mlxPerceptronClf')
    clf_pack.drop_clf('mlxMLP')
    clf_pack.drop_clf('mlxLogisticRegression')
    clf_pack.drop_clf('mlxAdaline')
    clf_pack.drop_clf('skLinear_SVC')

    train_Xs, train_Ys = train.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid.full_batch(['Xs', 'Ys'])
    score = clf_pack.score(train_Xs, train_Ys)
    pprint(score)

    score = clf_pack.score(valid_Xs, valid_Ys)
    pprint(score)
    #
    esm_pack = clf_pack.to_ensembleClfpack()
    train, valid = train_set.split((2, 7))
    train_Xs, train_Ys = train.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid.full_batch(['Xs', 'Ys'])

    esm_pack.fit(train_Xs, train_Ys)
    pprint(esm_pack.score_pack(train_Xs, train_Ys))
    pprint(esm_pack.score_pack(valid_Xs, valid_Ys))

    test_Xs = test_set.full_batch(['Xs'])

    predict = esm_pack.predict(test_Xs)['FoldingHardVote']
    # predict = clf_pack.predict(test_Xs)['skBagging']
    pprint(predict)
    pprint(predict.shape)
    submit_path = './submit.csv'
    datapack.to_kaggle_submit_csv(submit_path, predict)

    # clf_pack.dump(path)


def titanic_GAN_test():
    datapack = DatasetPackLoader().load_dataset('titanic')
    # datapack = DatasetPackLoader().load_dataset('titanic')
    train_set = datapack['train']
    test_set = datapack['test']
    train_set.shuffle()
    train, valid = train_set.split()

    gan = GAN()


def to_zero_one_encoding(x):
    x[x >= 0.5] = 1.0
    x[x < 0.5] = 0.0
    return x


def test_C_GAN():
    datapack = DatasetPackLoader().load_dataset('titanic')
    train_set = datapack['train']
    test_set = datapack['test']
    train_set.shuffle()

    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    sample_Xs, sample_Ys = train_Xs[:1], train_Ys[:2]

    gan = C_GAN()
    gan.train(train_Xs, train_Ys, epoch=1)

    metric = gan.metric(train_Xs, train_Ys)
    pprint(metric)

    gen = gan.generate(2, [[1, 0], [1, 0]])

    gen = to_zero_one_encoding(gen)
    pprint(gen)

    path = gan.save()
    gan = C_GAN()
    gan.load(path)
    gan.train(train_Xs, train_Ys, epoch=1)

    metric = gan.metric(train_Xs, train_Ys)
    pprint(metric)

    gen = gan.generate(2, [[1, 0], [1, 0]])
    gen = to_zero_one_encoding(gen)
    pprint(gen)


def exp_C_GAN_with_titanic():
    def show(data):
        pass

    pass
    #
    data_size = 4000
    zero_one_rate = 0.5
    datapack = DatasetPackLoader().load_dataset('titanic')
    train_set = datapack['train']
    test_set = datapack['test']
    train_set.shuffle()
    train, valid = train_set.split()

    train_Xs, train_Ys = train.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid.full_batch(['Xs', 'Ys'])
    #
    print(train_Xs.shape, train_Ys.shape)
    # path = 'C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\instance\\demetoir_C_GAN_2018-06-29_02-33-02'
    # if not os.path.exists(path):
    gan = C_GAN(learning_rate=0.001, n_noise=32, loss_type='GAN', with_clipping=True, clipping=.15)
    gan.train(train_Xs, train_Ys, epoch=1000)
    # path = gan.save()
    # print(path)

    # path = "C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\instance\\demetoir_C_GAN_2018-06-29_03-06-38"
    # gan = C_GAN().load(path)

    metric = gan.metric(train_Xs, train_Ys)
    pprint(metric)

    Ys_gen = [[1, 0] for _ in range(int(data_size * zero_one_rate))] \
             + [[0, 1] for _ in range(int(data_size * (1 - zero_one_rate)))]
    Ys_gen = np.array(Ys_gen)

    Xs_gen = gan.generate(1, Ys_gen[:1])
    Xs_gen = to_zero_one_encoding(Xs_gen)
    pprint(Xs_gen)
    pprint(Xs_gen.shape)

    # plot_1d(train_Xs[:1])
    # plot_1d(Xs_gen)
    # plt.plot(train_Xs[:1])
    # plt.show()

    # Xs_merge = np.concatenate([Xs_gen, train_Xs], axis=0)
    # Ys_merge = np.concatenate([Ys_gen, train_Ys], axis=0)
    # clf_pack = ClassifierPack()
    # # clf_pack.drop_clf('skQDA')
    # clf_pack.drop_clf('skGaussian_NB')
    # clf_pack.drop_clf('mlxSoftmaxRegressionClf')
    # clf_pack.drop_clf('mlxPerceptronClf')
    # clf_pack.drop_clf('mlxMLP')
    # clf_pack.drop_clf('mlxLogisticRegression')
    # clf_pack.drop_clf('mlxAdaline')
    # clf_pack.drop_clf('skLinear_SVC')
    # clf_pack.drop_clf('skSGD')
    # clf_pack.drop_clf('skRBF_SVM')
    # clf_pack.drop_clf('skMultinomial_NB')
    # clf_pack.fit(Xs_merge, Ys_merge)
    #
    # score = clf_pack.score(Xs_merge, Ys_merge)
    # pprint(score)
    #
    # score = clf_pack.score(Xs_gen, Ys_gen)
    # pprint(score)
    #
    # score = clf_pack.score(train_Xs, train_Ys)
    # pprint(score)
    #
    # score = clf_pack.score(valid_Xs, valid_Ys)
    # pprint(score)

    # esm_pack = clf_pack.to_ensembleClfpack()
    # esm_pack.fit(Xs_merge, Ys_merge)
    #
    # score = esm_pack.score_pack(Xs_gen, Ys_gen)
    # pprint(score)
    #
    # score = esm_pack.score_pack(train_Xs, train_Ys)
    # pprint(score)
    #
    # score = esm_pack.score_pack(valid_Xs, valid_Ys)
    # pprint(score)


def exp_CVAE_with_titanic():
    data_size = 4000
    zero_one_rate = 0.5
    datapack = DatasetPackLoader().load_dataset('titanic')
    train_set = datapack['train']
    test_set = datapack['test']
    train_set.shuffle()
    train, valid = train_set.split()

    train_Xs, train_Ys = train.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid.full_batch(['Xs', 'Ys'])

    # path = 'C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\instance\\demetoir_C_GAN_2018-06-29_02-33-02'
    # if not os.path.exists(path):
    cvae = CVAE(learning_rate=0.1, z_size=32, verbose=20, KL_D_rate=.05)
    cvae.train(train_Xs, train_Ys, epoch=1)
    # path = gan.save()
    # print(path)

    # path = "C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\instance\\demetoir_C_GAN_2018-06-29_03-06-38"
    # gan = C_GAN().load(path)

    metric = cvae.metric(train_Xs, train_Ys)
    pprint(metric)

    Xs_gen = cvae.recon(train_Xs, train_Ys)

    # plot_1d(Xs_gen[:1])
    # plot_1d(train_Xs[:1])

    #
    # Xs_gen = np.concatenate([Xs_gen, cvae.recon(train_Xs, train_Ys)], axis=0)
    # Xs_gen = np.concatenate([Xs_gen, cvae.recon(train_Xs, train_Ys)], axis=0)
    # Xs_gen = np.concatenate([Xs_gen, cvae.recon(train_Xs, train_Ys)], axis=0)
    # Xs_gen = np.concatenate([Xs_gen, cvae.recon(train_Xs, train_Ys)], axis=0)
    # Xs_gen = to_zero_one_encoding(Xs_gen)
    # pprint(Xs_gen)
    # pprint(Xs_gen.shape)
    # Ys_gen = np.concatenate([train_Ys, train_Ys, train_Ys, train_Ys, train_Ys], axis=0)

    # Xs_merge = np.concatenate([Xs_gen, train_Xs], axis=0)
    # Ys_merge = np.concatenate([Ys_gen, train_Ys], axis=0)
    # clf_pack = ClassifierPack()
    # # clf_pack.drop_clf('skQDA')
    # clf_pack.drop_clf('skGaussian_NB')
    # clf_pack.drop_clf('mlxSoftmaxRegressionClf')
    # clf_pack.drop_clf('mlxPerceptronClf')
    # clf_pack.drop_clf('mlxMLP')
    # clf_pack.drop_clf('mlxLogisticRegression')
    # clf_pack.drop_clf('mlxAdaline')
    # clf_pack.drop_clf('skLinear_SVC')
    # clf_pack.drop_clf('skSGD')
    # clf_pack.drop_clf('skRBF_SVM')
    # clf_pack.drop_clf('skMultinomial_NB')
    # clf_pack.fit(Xs_gen, Ys_gen)
    #
    # score = clf_pack.score(Xs_gen, Ys_gen)
    # pprint(score)
    #
    # score = clf_pack.score(Xs_gen, Ys_gen)
    # pprint(score)
    #
    # score = clf_pack.score(train_Xs, train_Ys)
    # pprint(score)
    #
    # score = clf_pack.score(valid_Xs, valid_Ys)
    # pprint(score)
    #
    # esm_pack = clf_pack.to_ensembleClfpack()
    # esm_pack.fit(Xs_gen, Ys_gen)
    #
    # score = esm_pack.score(Xs_gen, Ys_gen)
    # pprint(score)
    #
    # score = esm_pack.score(train_Xs, train_Ys)
    # pprint(score)
    #
    # score = esm_pack.score(valid_Xs, valid_Ys)
    # pprint(score)
    #
    # test_Xs = test_set.full_batch('Xs')
    # predict = esm_pack.predict(test_Xs)['FoldingHardVote']
    # # predict = clf_pack.predict(test_Xs)['skBagging']
    # pprint(predict)
    # pprint(predict.shape)
    # submit_path = './submit.csv'
    # datapack.to_kaggle_submit_csv(submit_path, predict)


def plt_common(plt, fig, path=None, show=False, title=None, extend='.png'):
    if title is None:
        title = time_stamp()
    plt.title(title)

    if path is None:
        path = path_join('.', 'matplot', title + extend)
    setup_file(path)
    fig.savefig(path)

    if show:
        plt.show()


def plot_line(x, path=None, show=False, title=None, extend='.png'):
    fig = plt.figure()

    plt.plot(x)

    plt_common(plt, fig, path=path, show=show, title=title, extend=extend)


def plot_dot(*args, path=None, show=False, title=None, extend='.png'):
    sns.set()
    sns.set_style('whitegrid')
    sns.set_color_codes()

    current_palette = sns.color_palette()
    print(current_palette)
    sns.palplot(current_palette)

    fig = plt.figure()

    color_set = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
    marker_set = ['.', ',', 'o', 'v', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd',
                  '|', '_', ]
    line_set = ['-', '--', '-.', ':']
    """
        ``'-'``          solid line style
        ``'--'``         dashed line style
        ``'-.'``         dash-dot line style
        ``':'``          dotted line style
    """

    """
        ``'.'``          point marker
        ``','``          pixel marker
        ``'o'``          circle marker
        ``'v'``          triangle_down marker
        ``'^'``          triangle_up marker
        ``'<'``          triangle_left marker
        ``'>'``          triangle_right marker
        ``'1'``          tri_down marker
        ``'2'``          tri_up marker
        ``'3'``          tri_left marker
        ``'4'``          tri_right marker
        ``'s'``          square marker
        ``'p'``          pentagon marker
        ``'*'``          star marker
        ``'h'``          hexagon1 marker
        ``'H'``          hexagon2 marker
        ``'+'``          plus marker
        ``'x'``          x marker
        ``'D'``          diamond marker
        ``'d'``          thin_diamond marker
        ``'|'``          vline marker
        ``'_'``          hline marker
    """

    new_xs = []
    for x in args:
        new_xs += [x, 'b--']
    xs = new_xs

    plt.plot(*xs)

    plt_common(plt, fig, path=path, show=show, title=title, extend=extend)


def hist(x, path=None, show=False, title=None, extend='.png'):
    fig = plt.figure()

    plt.hist(x)

    plt_common(plt, fig, path=path, show=show, title=title, extend=extend)


def scatter(x, y, path=None, show=False, title=None, extend='.png'):
    fig = plt.figure(x, y)

    plt.scatter(x)

    plt_common(plt, fig, path=path, show=show, title=title, extend=extend)


class plot_tools:
    pass


def test_plot_tools():
    x = np.array([-i for i in range(32)])
    y = np.array([i for i in range(32)])
    z = np.array([i * 2 for i in range(32)])
    w = np.array([i * 4 for i in range(32)])
    v = np.array([i * 6 for i in range(32)])
    k = np.array([i * 8 for i in range(32)])

    # plot_line(x)
    xs = [np.array([i * k for i in range(32)]) for k in range(-10, 10)]

    plot_dot(*xs)
    # plot_1d(x)
    #
    # hist(x)

    # scatter(x, y)


@deco_timeit
def main():
    # titanic_submit()
    # test_C_GAN()
    # exp_C_GAN_with_titanic()
    # exp_CVAE_with_titanic()

    # s = '<>1234sp*hH+xDd|_'
    # for c in s:
    #     bprint(f""" '{c}', """, end=' ')
    test_plot_tools()

    pass
