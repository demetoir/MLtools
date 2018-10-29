# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from keras import Input, Model, optimizers
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, ReLU, Softmax

from script.data_handler.MNIST import MNIST
from script.model.sklearn_like_model.BaseModel import BaseModel
from script.model.sklearn_like_model.NetModule.BaseNetModule import BaseNetModule
from script.model.sklearn_like_model.NetModule.optimizer.Adam import Adam
from script.util.Logger import pprint_logger, Logger
from script.util.PlotTools import PlotTools
from script.util.deco import deco_timeit
from slackbot.SlackBot import deco_slackbot

bprint = print
logger = Logger('bench_code', level='INFO', )
print = logger.info
pprint = pprint_logger(print)
NpArr = np.array
DF = pd.DataFrame
Series = pd.Series
plot = PlotTools(save=True, show=False)


def build_model(DropoutRatio=0.5):
    capacity = 16
    input_layer = Input((28, 28, 1))
    # 28 * 28
    conv1 = Conv2D(32, (3, 3), padding="same")(input_layer)
    conv1 = ReLU()(conv1)

    # 14*14
    conv1 = Conv2D(64, (3, 3), padding="same")(conv1)
    conv1 = ReLU()(conv1)

    conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    flatten = Flatten()(conv1)

    dense = Dense(128)(flatten)
    dense = ReLU()(dense)

    dense = Dense(10)(dense)
    softmax = Softmax()(dense)

    model = Model(input_layer, softmax)

    return model


def test_keras_on_mnist():
    mnist = MNIST()
    mnist.load(path=f'./data/MNIST')
    train_set = mnist.train
    test_set = mnist.test

    print(mnist)
    # return
    print(train_set)
    print(test_set)

    x_train, y_train = train_set.full_batch()
    x_test, y_test = test_set.full_batch()

    model = build_model()
    model.summary()
    adam = optimizers.adam(lr=0.01)
    # adam = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    # adam = optimizers.Nadam(lr=0.01)
    model.compile(
        loss="categorical_crossentropy", optimizer=adam,
        metrics=['accuracy']
    )

    tensorboard = TensorBoard(
        log_dir='./tf_summary/keras/test_mnist_Nadam', histogram_freq=0, batch_size=32, write_graph=True,
        write_grads=False, write_images=False, embeddings_freq=0,
        embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
        update_freq='epoch'
    )

    model.fit(
        x=x_train,
        y=y_train,
        validation_data=[x_test, y_test],
        batch_size=32,
        epochs=10,
        # callbacks=[tensorboard]
    )

    # cnn
    # dense

    # keras model
    # adam
    # NAG
    # SGD

    # my model
    # adam
    # NAG
    # SGD


from script.util.tensor_ops import *


class Cnn(BaseNetModule):
    def __init__(self, x, capacity=None, reuse=False, name=None, verbose=0):
        super().__init__(capacity, reuse, name, verbose)
        self.x = x
        if capacity:
            self.n_channel = capacity
        else:
            self.n_channel = 64

    def build(self):
        with tf.variable_scope(self.name):
            # stack = Stacker(self.x)
            # stack.add_layer(Conv2d(32, (3, 3), padding='SAME'))
            # stack.add_layer(Relu())
            #
            # stack.add_layer(Conv2d(64, (3, 3), padding='SAME'))
            # stack.add_layer(Relu())
            #
            # stack.add_layer(MaxPooling2d((2, 2)))
            # stack.flatten()
            # stack.add_layer(Linear(128))
            # stack.add_layer(Relu())
            #
            # logit = stack.add_layer(Linear(10))
            # softmax = stack.add_layer(Softmax())
            #
            # self.logit = logit
            # self.softmax = softmax

            conv = Conv2d(32, (3, 3), padding='SAME')(self.x)
            conv = Relu()(conv)

            # 14*14
            conv = Conv2d(64, (3, 3), padding='SAME')(conv)
            conv = Relu()(conv)

            conv = MaxPooling2d((2, 2))(conv)
            conv = flatten(conv)

            dense = Linear(128)(conv)
            dense = Relu()(dense)

            logit = Linear(10)(dense)
            softmax = Softmax()(logit)

            self.logit = logit
            self.softmax = softmax


class SimpleModel(BaseModel):
    def __init__(
            self,
            verbose=10,
            **kwargs
    ):
        BaseModel.__init__(self, verbose, **kwargs)

    def _build_main_graph(self):
        cnn = Cnn(self.Xs)
        cnn.build()
        self._logit = cnn.logit
        self._proba = cnn.softmax
        self.vars = cnn.vars

    def _build_loss_ops(self):
        def softmax_cross_entropy(trues, logits, mean=True):
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=trues, logits=logits)
            if mean:
                loss = tf.reduce_mean(loss)
            return loss

        self.loss_ops = softmax_cross_entropy(self.Ys, self._logit)
        return [self.loss_ops]

    def _build_train_ops(self):
        self.optimizer = Adam(0.01).minimize(self.loss_ops, self.vars).build()
        return [self.optimizer.train_op]

    def _build_metric_ops(self):
        def accuracy(true, predict, mean=True):
            acc = tf.cast(tf.equal(true, predict), tf.float64, name="acc")
            if mean:
                acc = tf.reduce_mean(acc)
            return acc

        self.Ys_label = onehot_to_index(self.Ys)
        self.acc = accuracy(self.Ys_label, self._predict)
        return [self.acc]

    def _build_predict_ops(self):
        def categorical_predict(proba):
            return tf.cast(tf.argmax(proba, 1, name="predicted_label"), tf.float32)

        self._predict = categorical_predict(self._proba)
        return [self._predict]


def test_my_model():
    mnist = MNIST()
    mnist.load(path=f'./data/MNIST')
    train_set = mnist.train
    test_set = mnist.test

    print(mnist)
    # return
    print(train_set)
    print(test_set)

    x_train, y_train = train_set.full_batch()
    x_test, y_test = test_set.full_batch()

    model = SimpleModel()
    model.build(x=(28, 28, 1), y=(10,))
    model.train(
        x_train,
        y_train,
        epoch_callbacks=[],
        epoch=10,
        batch_size=32,
        # validation_data=[x_test, y_test]
    )


@deco_timeit
@deco_slackbot('./slackbot/tokens/ml_bot_token', 'mltool_bot')
def main():
    # from tqdm import trange
    # from random import random, randint
    # from time import sleep
    #
    # epoch = 100
    # with trange(100) as t:
    #     for i in t:
    #         # Description will be displayed on the left
    #         t.set_description(f'epoch {i}/{epoch}')
    #         # Postfix will be displayed on the right,
    #         # formatted automatically based on argument's datatype
    #         t.set_postfix(
    #             loss=random(), gen=randint(1, 999))
    #         sleep(0.1)
    #
    # with tqdm(total=100, bar_format="{postfix[0]} {postfix[1][value]:>8.2g}",
    #           postfix=["Batch", dict(value=0)]) as t:
    #     for i in range(100):
    #         sleep(0.1)
    #         t.postfix[1]["value"] = i / 2
    #         t.update()

    test_my_model()
    # test_keras_on_mnist()
    # ss = SS_baseline()
    # ss.new_model()
    # ss.train()

    # train_kernel_main()
    pass
