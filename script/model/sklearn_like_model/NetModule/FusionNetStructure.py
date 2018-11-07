from script.model.sklearn_like_model.NetModule.BaseNetModule import BaseNetModule
from script.model.sklearn_like_model.NetModule.DynamicDropoutRate import DynamicDropoutRate
from script.util.Stacker import Stacker
from script.util.tensor_ops import *


def _residual_block_pre_activation(x, n_channel, filter_=(3, 3), name='residual_block_pre_activation'):
    with tf.variable_scope(name):
        stack = Stacker(x)

        stack.add_layer(batchActivation)
        stack.layers_conv2d(n_channel, filter_, (1, 1), 'SAME')

        stack.add_layer(batchActivation)
        stack.layers_conv2d(n_channel, filter_, (1, 1), 'SAME')

        stack.residual_add(x)

        return stack.last_layer


def _residual_atrous_block(x, n_channel, filter_, activation, name='residual_block'):
    with tf.variable_scope(name):
        x_in = x
        stack = Stacker(x)
        stack.atrous_conv2d_block(n_channel, filter_, 2, activation)
        # stack.atrous_conv2d_block(n_channel, filter_, 2, activation)
        stack.atrous_conv2d_block(n_channel, filter_, 2, activation)
        stack.residual_add(x_in)

    return stack.last_layer


def batchActivation(x):
    x = BN()(x)
    x = Relu()(x)
    return x


class FusionNetModule(BaseNetModule):
    def __init__(
            self,
            x,
            level=4,
            n_classes=2,
            depth=1,
            capacity=64,
            dropout_rate=0.5,
            reuse=False,
            name=None,
            verbose=0
    ):
        super().__init__(capacity, reuse, name, verbose)

        self.x = x
        self.level = level
        self.n_classes = n_classes
        self.depth = depth
        self.n_channel = capacity
        self.dropout_rate = dropout_rate
        self.dropout_rate = 0.5

    def build(self):
        self.DynamicDropoutRate = DynamicDropoutRate(self.dropout_rate)
        self.drop_out_tensor = self.DynamicDropoutRate.tensor
        self.dropout_rate_tensor = self.drop_out_tensor

        with tf.variable_scope(self.name, reuse=self.reuse):
            self.skip_connects = []
            self.stacker = Stacker(self.x, name='')
            stacker = self.stacker

            # 101 to 51
            stacker.layers_conv2d(self.n_channel, (3, 3), (1, 1), 'SAME')
            stacker.add_layer(batchActivation)
            stacker.add_layer(_residual_block_pre_activation, self.n_channel)
            stacker.add_layer(_residual_block_pre_activation, self.n_channel)
            stacker.add_layer(batchActivation)
            self.skip_connects += [stacker.last_layer]
            stacker.layers_max_pooling2d((2, 2), (2, 2))
            stacker.add_layer(Dropout(self.dropout_rate / 2))

            # 51 to 25
            stacker.layers_conv2d(self.n_channel * 2, (3, 3), (1, 1), 'SAME')
            stacker.add_layer(batchActivation)
            stacker.add_layer(_residual_block_pre_activation, self.n_channel * 2)
            stacker.add_layer(_residual_block_pre_activation, self.n_channel * 2)
            stacker.add_layer(batchActivation)
            self.skip_connects += [stacker.last_layer]
            stacker.layers_max_pooling2d((2, 2), (2, 2))
            stacker.add_layer(Dropout(self.dropout_rate))

            # 25 to 13
            stacker.layers_conv2d(self.n_channel * 4, (3, 3), (1, 1), 'SAME')
            stacker.add_layer(batchActivation)
            stacker.add_layer(_residual_block_pre_activation, self.n_channel * 4)
            stacker.add_layer(_residual_block_pre_activation, self.n_channel * 4)
            stacker.add_layer(batchActivation)
            self.skip_connects += [stacker.last_layer]
            stacker.layers_max_pooling2d((2, 2), (2, 2))
            stacker.add_layer(Dropout(self.dropout_rate))

            # 13 to 7
            stacker.layers_conv2d(self.n_channel * 8, (3, 3), (1, 1), 'SAME')
            stacker.add_layer(batchActivation)
            stacker.add_layer(_residual_block_pre_activation, self.n_channel * 8)
            stacker.add_layer(_residual_block_pre_activation, self.n_channel * 8)
            stacker.add_layer(batchActivation)
            self.skip_connects += [stacker.last_layer]
            stacker.layers_max_pooling2d((2, 2), (2, 2))
            stacker.add_layer(Dropout(self.dropout_rate))

            print('middle')
            # middle
            stacker.layers_conv2d(self.n_channel * 16, (3, 3), (1, 1), 'SAME')
            stacker.add_layer(batchActivation)
            stacker.add_layer(_residual_block_pre_activation, self.n_channel * 16)
            stacker.add_layer(_residual_block_pre_activation, self.n_channel * 16)
            stacker.add_layer(batchActivation)

            skip_connects = self.skip_connects[::-1]

            # 7 to 13
            stacker.layers_conv2d_transpose(self.n_channel * 8, (3, 3), (2, 2), 'SAME')
            stacker.add_layer(batchActivation)
            stacker.concat([stacker.last_layer, skip_connects[0]], 3)
            stacker.add_layer(Dropout(self.dropout_rate))
            stacker.layers_conv2d(self.n_channel * 8, (3, 3), (1, 1), 'SAME')
            stacker.add_layer(batchActivation)
            stacker.add_layer(_residual_block_pre_activation, self.n_channel * 8)
            stacker.add_layer(_residual_block_pre_activation, self.n_channel * 8)
            stacker.add_layer(batchActivation)

            # 12 to 25
            stacker.layers_conv2d_transpose(self.n_channel * 4, (3, 3), (2, 2), 'VALID')
            stacker.add_layer(batchActivation)
            stacker.concat([stacker.last_layer, skip_connects[1]], 3)
            stacker.add_layer(Dropout(self.dropout_rate))
            stacker.layers_conv2d(self.n_channel * 4, (3, 3), (1, 1), 'SAME')
            stacker.add_layer(batchActivation)
            stacker.add_layer(_residual_block_pre_activation, self.n_channel * 4)
            stacker.add_layer(_residual_block_pre_activation, self.n_channel * 4)
            stacker.add_layer(batchActivation)

            # 25 to 50
            stacker.layers_conv2d_transpose(self.n_channel * 2, (3, 3), (2, 2), 'SAME')
            stacker.add_layer(batchActivation)
            stacker.concat([stacker.last_layer, skip_connects[2]], 3)
            stacker.add_layer(Dropout(self.dropout_rate))
            stacker.layers_conv2d(self.n_channel * 2, (3, 3), (1, 1), 'SAME')
            stacker.add_layer(batchActivation)
            stacker.add_layer(_residual_block_pre_activation, self.n_channel * 2)
            stacker.add_layer(_residual_block_pre_activation, self.n_channel * 2)
            stacker.add_layer(batchActivation)

            # 50 to 101
            stacker.layers_conv2d_transpose(self.n_channel * 1, (3, 3), (2, 2), 'VALID')
            stacker.add_layer(batchActivation)
            stacker.concat([stacker.last_layer, skip_connects[3]], 3)
            stacker.add_layer(Dropout(self.dropout_rate))
            stacker.layers_conv2d(self.n_channel * 1, (3, 3), (1, 1), 'SAME')
            stacker.add_layer(batchActivation)
            stacker.add_layer(_residual_block_pre_activation, self.n_channel * 1)
            stacker.add_layer(_residual_block_pre_activation, self.n_channel * 1)
            stacker.add_layer(batchActivation)

            self.logit = stacker.layers_conv2d(self.n_classes, (1, 1), (1, 1), 'SAME')
            if self.n_classes == 1:
                self.proba = stacker.sigmoid()
            else:
                self.proba = stacker.pixel_wise_softmax()

        return self

    def set_train(self, sess):
        self.DynamicDropoutRate.set_train(sess)

    def set_non_train(self, sess):
        self.DynamicDropoutRate.set_non_train(sess)

    def update_dropout_rate(self, sess, x):
        self.dropout_rate = x
        self.DynamicDropoutRate.update(sess, x)
