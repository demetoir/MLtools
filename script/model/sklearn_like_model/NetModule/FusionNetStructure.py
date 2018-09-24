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

    @staticmethod
    def conv_seq(stacker, n_channel, depth, dropout_rate_tensor):
        stacker.conv_block(n_channel, CONV_FILTER_3311, relu)

        for i in range(depth):
            stacker.add_layer(_residual_block, n_channel, CONV_FILTER_3311, relu)
            if dropout_rate_tensor:
                stacker.dropout(dropout_rate_tensor)

        stacker.conv_block(n_channel, CONV_FILTER_3311, relu)
        return stacker

    def encoder(self, x, n_channel, depth, level, dropout_rate_tensor):
        def recursive(stacker, n_channel, level):
            if level is 0:
                return []

            stacker = self.conv_seq(stacker, n_channel, depth, dropout_rate_tensor)
            skip_tensor = stacker.last_layer
            stacker.max_pooling(CONV_FILTER_2222)

            return [skip_tensor] + recursive(stacker, n_channel * 2, level - 1)

        with tf.variable_scope('encoder'):
            self.encoder_stacker = Stacker(x)
            self.skip_connects = recursive(self.encoder_stacker, n_channel, level)

        return self.encoder_stacker.last_layer, self.skip_connects

    def decoder(self, x, skip_connects, n_channel, depth, level, dropout_rate_tensor):
        def recursive(stacker, n_channel, level):
            if level is 0:
                return

            recursive(stacker, n_channel * 2, level - 1)

            stacker.upscale_2x_block(n_channel, CONV_FILTER_2211, relu)

            # TODO hack to dynamic batch size after conv transpose, concat must need. wtf?
            skip_tensor = skip_connects[level - 1]
            stacker.concat(skip_tensor, axis=3)

            stacker.conv_block(n_channel, CONV_FILTER_1111, relu)
            stacker.residual_add(skip_tensor)

            self.conv_seq(stacker, n_channel, depth, dropout_rate_tensor)

        skip_connects = skip_connects[::-1]
        with tf.variable_scope('decoder'):
            self.decoder_stacker = Stacker(x)
            recursive(self.decoder_stacker, n_channel, level)

        return self.decoder_stacker.last_layer

    def bottom_layer(self, x, n_channel, depth, dropout_rate_tensor):
        with tf.variable_scope('bottom_layer'):
            stacker = Stacker(x)
            stacker = self.conv_seq(stacker, n_channel, depth, dropout_rate_tensor)
            self.bottom_layer_stacker = stacker

        return self.bottom_layer_stacker.last_layer

    def build(self):

        self.DynamicDropoutRate = DynamicDropoutRate(self.dropout_rate)
        self.drop_out_tensor = self.DynamicDropoutRate.tensor

        def recursive_build(stacker, n_channel, level, dropout_rate_tensor=None):
            if level == 0:
                stacker.conv_block(n_channel, CONV_FILTER_3311, relu)

                for i in range(self.depth):
                    stacker.add_layer(_residual_block, n_channel, CONV_FILTER_3311, relu)
                    if dropout_rate_tensor:
                        stacker.dropout(dropout_rate_tensor)

                stacker.conv_block(n_channel, CONV_FILTER_3311, relu)

            else:
                # encode
                stacker.conv_block(n_channel, CONV_FILTER_3311, relu)

                for i in range(self.depth):
                    stacker.add_layer(_residual_block, n_channel, CONV_FILTER_3311, relu)
                    if dropout_rate_tensor:
                        stacker.dropout(dropout_rate_tensor)

                stacker.conv_block(n_channel, CONV_FILTER_3311, relu)

                x_add = stacker.last_layer
                concat = stacker.last_layer

                stacker.max_pooling(CONV_FILTER_2222)

                # recursion
                stacker = recursive_build(stacker, n_channel * 2, level - 1)

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

        with tf.variable_scope(self.name, reuse=self.reuse):
            recursive = False
            if recursive:
                self.stacker = Stacker(self.x, verbose=self.verbose)
                stacker = recursive_build(self.stacker, self.n_channel, self.level, self.drop_out_tensor)

                self.logit = stacker.conv_block(self.n_classes, CONV_FILTER_3311, relu)
                stacker.pixel_wise_softmax()
                self.proba = stacker.last_layer
            else:
                encode, skip_tensors = self.encoder(
                    self.x, self.n_channel, self.depth, self.level,
                    self.drop_out_tensor)

                bottom_layer = self.bottom_layer(
                    encode,
                    self.n_channel * (2 ** (self.level - 1)),
                    self.depth,
                    self.drop_out_tensor
                )

                decode = self.decoder(
                    bottom_layer,
                    skip_tensors,
                    self.n_channel,
                    self.depth,
                    self.level,
                    self.drop_out_tensor,
                )

                self.decoder_stacker.conv_block(self.n_classes, CONV_FILTER_3311, relu)
                self.logit = self.decoder_stacker.last_layer
                self.proba = pixel_wise_softmax(self.logit)

    def set_train(self, sess):
        self.DynamicDropoutRate.set_train(sess)

    def set_predict(self, sess):
        self.DynamicDropoutRate.set_predict(sess)

    def update_dropout_rate(self, sess, x):
        self.dropout_rate = x
        self.DynamicDropoutRate.update(sess, x)
