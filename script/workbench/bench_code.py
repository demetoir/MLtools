# -*- coding:utf-8 -*-
from script.model.sklearn_like_model.MLPClassifier import MLPClassifier
from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.model.sklearn_like_model.AE.AutoEncoder import AutoEncoder
from script.sklearn_like_toolkit.ClassifierPack import ClassifierPack
from script.model.sklearn_like_model.GAN.GAN import GAN
from script.util.Logger import pprint_logger, Logger
import numpy as np
from script.util.deco import deco_timeit

########################################################################################################################
# print(built-in function) is not good for logging

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


def test_GAN():
    class_ = GAN
    data_pack = DatasetPackLoader().load_dataset("titanic")
    dataset = data_pack['train']
    Xs, Ys = dataset.full_batch(['Xs', 'Ys'])
    sample_X = Xs[:2]
    sample_Y = Ys[:2]

    model = class_()
    model.train(Xs, epoch=1)

    metric = model.metric(sample_X)
    print(metric)

    gen = model.generate(size=2)
    print(gen)

    path = model.save()

    model = class_()
    model.load(path)
    print('model reloaded')

    for i in range(2):
        model.train(Xs, epoch=1)

    metric = model.metric(sample_X)
    print(metric)

    model.save()


def test_AE():
    class_ = AutoEncoder
    data_pack = DatasetPackLoader().load_dataset("titanic")
    dataset = data_pack['train']
    Xs, Ys = dataset.full_batch(['Xs', 'Ys'])
    sample_X = Xs[:2]
    sample_Y = Ys[:2]

    model = class_()
    model.train(Xs, epoch=1)

    metric = model.metric(sample_X)
    print(metric)

    code = model.code(sample_X)
    print(code)

    recon = model.recon(sample_X)
    print(recon)

    path = model.save()

    model = class_()
    model.load(path)
    print('model reloaded')

    for i in range(2):
        model.train(Xs, epoch=1)

    metric = model.metric(sample_X)
    print(metric)

    metric = model.metric(sample_X)
    print(metric)

    code = model.code(sample_X)
    print(code)

    recon = model.recon(sample_X)
    print(recon)

    model.save()


def test_CLf():
    class_ = MLPClassifier
    data_pack = DatasetPackLoader().load_dataset("titanic")
    dataset = data_pack['train']
    Xs, Ys = dataset.full_batch(['Xs', 'Ys'])
    sample_X = Xs[:2]
    sample_Y = Ys[:2]

    model = class_()
    model.train(Xs, Ys)

    predict = model.predict(sample_X)
    print(predict)

    score = model.score(Xs, Ys)
    print(score)

    proba = model.proba(sample_X)
    print(proba)

    metric = model.metric(sample_X, sample_Y)
    print(metric)

    path = model.save()

    model = class_()
    model.load(path)
    model.train(Xs, Ys)

    predict = model.predict(sample_X)
    print(predict)

    score = model.score(Xs, Ys)
    print(score)

    proba = model.proba(sample_X)
    print(proba)

    metric = model.metric(sample_X, sample_Y)
    print(metric)

    model.save()


@deco_timeit
def main():
    test_CLf()
    # test_GAN()
    # test_AE()
