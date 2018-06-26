# -*- coding:utf-8 -*-
from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.model.sklearn_like_model.AE.AutoEncoder import AutoEncoder
# from script.sklearn_like_toolkit.ClassifierPack import ClassifierPack
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


@deco_timeit
def main():
    class_ = GAN
    data_pack = DatasetPackLoader().load_dataset("titanic")
    dataset = data_pack['train']
    Xs, Ys = dataset.full_batch(['Xs', 'Ys'])
    sample_X = Xs[:2]
    sample_Y = Ys[:2]

    # model = class_(dataset.input_shapes)
    # model.build()
    #
    # model.train(Xs, epoch=1)
    #
    # metric = model.metric(sample_X)
    # print(metric)
    #
    # gen = model.generate(size=2)
    # print(gen)
    #
    # path = model.save()
    # path = """C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\instance\\demetoir_GAN_2018-06-26_06-52-36"""
    path = None
    model = class_()
    model.load(path)
    print('model reloaded')

    for i in range(100):
        model.train(Xs, epoch=1)

    gen = model.generate(size=1)

    gen[gen < 0.5] = 0
    gen[gen > 0.5] = 1
    print(gen)

    metric = model.metric(sample_X)
    print(metric)

    model.save()
