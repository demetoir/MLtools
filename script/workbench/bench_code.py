# -*- coding:utf-8 -*-
# from script.model.sklearn_like_model.AE.CVAE import CVAE
import os
from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.sklearn_like_toolkit.ClassifierPack import ClassifierPack
from script.util.Logger import pprint_logger, Logger
from script.util.deco import deco_timeit
import numpy as np

########################################################################################################################
# print(built-in function) is not good for logging
from script.util.misc_util import path_join, time_stamp, setup_file
from script.util.PlotTools import PlotTools

bprint = print
logger = Logger('bench_code', level='INFO', )
print = logger.info
pprint = pprint_logger(print)


#######################################################################################################################


def onehot_label_smooth(Ys, smooth=0.05):
    Ys *= (1 - smooth * 2)
    Ys += smooth
    return Ys


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


def to_zero_one_encoding(x):
    x[x >= 0.5] = 1.0
    x[x < 0.5] = 0.0
    return x



def test_plot_tool_async():
    rand_x_1d = np.random.normal(3, 5, [10])

    tool = PlotTools()
    for i in range(10):
        tool.dist(rand_x_1d)


@deco_timeit
def main():
    test_plot_tool_async()
    # titanic_submit()
    # test_C_GAN()
    # exp_C_GAN_with_titanic()
    # exp_CVAE_with_titanic()
    pass
