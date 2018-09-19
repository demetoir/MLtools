from script.model.sklearn_like_model.DynamicDropoutRate import DynamicDropoutRate
from script.model.sklearn_like_model.NetModule.BaseNetModule import BaseNetModule
from script.util.Stacker import Stacker
from script.util.tensor_ops import *


def _residual_block(x, n_channel, filter_, activation, name='residual_block'):
    with tf.variable_scope(name):
        x_in = x
        x = conv_block(x, n_channel, filter_, activation, name='conv_block1')
        x = conv_block(x, n_channel, filter_, activation, name='conv_block2')
        x = conv_block(x, n_channel, filter_, activation, name='conv_block3')
        x = residual_add(x, x_in)
    return x


class FusionNetModule(BaseNetModule):
    def __init__(self, x, level=4, n_classes=2, depth=1, capacity=64, dropout_rate=0.5, reuse=False, name=None,
                 verbose=0):
        super().__init__(capacity, reuse, name, verbose)

        self.x = x
        self.level = level
        self.n_classes = n_classes
        self.depth = depth
        self.n_channel = capacity
        self.dropout_rate = dropout_rate

    def build(self):
        self.DynamicDropoutRate = DynamicDropoutRate(self.dropout_rate)
        self.drop_out_tensor = self.DynamicDropoutRate.tensor

        def recursion(stacker, n_channel, level, dropout_rate_tensor=None):
            if level == 0:
                stacker.conv_block(n_channel, CONV_FILTER_3311, relu)
                # if dropout_rate_tensor:
                #     stacker.dropout(dropout_rate_tensor)

                for i in range(self.depth):
                    stacker.add_layer(_residual_block, n_channel, CONV_FILTER_3311, relu)
                    if dropout_rate_tensor:
                        stacker.dropout(dropout_rate_tensor)

                stacker.conv_block(n_channel, CONV_FILTER_3311, relu)
                # if dropout_rate_tensor:
                #     stacker.dropout(dropout_rate_tensor)

            else:
                # encode
                stacker.conv_block(n_channel, CONV_FILTER_3311, relu)
                # if dropout_rate_tensor:
                #     stacker.dropout(dropout_rate_tensor)

                for i in range(self.depth):
                    stacker.add_layer(_residual_block, n_channel, CONV_FILTER_3311, relu)
                    if dropout_rate_tensor:
                        stacker.dropout(dropout_rate_tensor)

                stacker.conv_block(n_channel, CONV_FILTER_3311, relu)
                # if dropout_rate_tensor:
                #     stacker.dropout(dropout_rate_tensor)

                x_add = stacker.last_layer
                concat = stacker.last_layer

                stacker.max_pooling(CONV_FILTER_2222)

                # recursion
                stacker = recursion(stacker, n_channel * 2, level - 1)

                # decode
                stacker.upscale_2x_block(n_channel, CONV_FILTER_2211, relu)

                # TODO hack to dynamic batch size after conv transpose, concat must need. wtf?
                stacker.concat(concat, axis=3)
                stacker.conv_block(n_channel, CONV_FILTER_1111, relu)

                stacker.residual_add(x_add)

                stacker.conv_block(n_channel, CONV_FILTER_3311, relu)
                # if dropout_rate_tensor:
                #     stacker.dropout(dropout_rate_tensor)

                for i in range(self.depth):
                    stacker.add_layer(_residual_block, n_channel, CONV_FILTER_3311, relu)
                    if dropout_rate_tensor:
                        stacker.dropout(dropout_rate_tensor)

                stacker.conv_block(n_channel, CONV_FILTER_3311, relu)
                # if dropout_rate_tensor:
                    # stacker.dropout(dropout_rate_tensor)

            return stacker

        self.stacker = Stacker(self.x, verbose=self.verbose)
        with tf.variable_scope(self.name, reuse=self.reuse):
            stacker = recursion(self.stacker, self.n_channel, self.level, self.drop_out_tensor)

            stacker.conv_block(self.n_classes, CONV_FILTER_3311, relu)
            self.logit = stacker.last_layer
            stacker.pixel_wise_softmax()
            self.proba = stacker.last_layer

    def set_train(self, sess):
        self.DynamicDropoutRate.set_train(sess)

    def set_predict(self, sess):
        self.DynamicDropoutRate.set_predict(sess)

    def update_dropout_rate(self, sess, x):
        self.dropout_rate = x
        self.DynamicDropoutRate.update(sess, x)
