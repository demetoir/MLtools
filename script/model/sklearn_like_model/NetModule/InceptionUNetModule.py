from script.model.sklearn_like_model.DynamicDropoutRate import DynamicDropoutRate
from script.model.sklearn_like_model.NetModule.BaseNetModule import BaseNetModule
from script.model.sklearn_like_model.NetModule.InceptionSructure.BaseInceptionNetModule import BaseInceptionNetModule
from script.util.Stacker import Stacker
from script.util.tensor_ops import *


def residual(stacker, module, channel):
    x_add = stacker.last_layer
    stacker.add_layer(module, channel)
    stacker.residual_add(x_add)
    return stacker


class InceptionV2UNetEncoderModule(BaseInceptionNetModule):
    @staticmethod
    def module_A(x, channels, name='module_A'):
        with tf.variable_scope(name):
            a = conv_block(x, channels['a0'], CONV_FILTER_1111, relu, name='branch_a0')

            b = conv_block(x, channels['b0'], CONV_FILTER_1111, relu, name='branch_b0')
            b = conv_block(b, channels['b1'], CONV_FILTER_5511, relu, name='branch_b1')

            c = conv_block(x, channels['c0'], CONV_FILTER_1111, relu, name='branch_c0')
            c = conv_block(c, channels['c1'], CONV_FILTER_3311, relu, name='branch_c1')
            c = conv_block(c, channels['c2'], CONV_FILTER_3311, relu, name='branch_c2')

            d = avg_pooling(x, CONV_FILTER_3311, name='branch_d0')
            d = conv_block(d, channels['d1'], CONV_FILTER_1111, relu, name='branch_d1')

            return concat([a, b, c, d], axis=3)

    @staticmethod
    def multi_pool_A(x, channels, name='multi_pool_A'):
        with tf.variable_scope(name):
            a = conv_block(x, channels['a0'], CONV_FILTER_3322, relu, name='branch_a0')

            b = conv_block(x, channels['b0'], CONV_FILTER_1111, relu, name='branch_b0')
            b = conv_block(b, channels['b1'], CONV_FILTER_3311, relu, name='branch_b1')
            b = conv_block(b, channels['b2'], CONV_FILTER_3322, relu, name='branch_b2')

            c = max_pooling(x, CONV_FILTER_3322, name='branch_c0')

            return concat([a, b, c], axis=3)

    @staticmethod
    def module_B(x, channels, name='module_B'):
        # mixed4: 17 x 17 x 768.

        with tf.variable_scope(name):
            a = conv_block(x, channels['a0'], CONV_FILTER_1111, relu, name='branch_a0')

            b = conv_block(x, channels['b0'], CONV_FILTER_1111, relu, name='branch_b0')
            b = conv_block(b, channels['b1'], CONV_FILTER_7111, relu, name='branch_b1')
            b = conv_block(b, channels['b2'], CONV_FILTER_1711, relu, name='branch_b2')

            c = conv_block(x, channels['c0'], CONV_FILTER_1111, relu, name='branch_c0')
            c = conv_block(c, channels['c1'], CONV_FILTER_7111, relu, name='branch_c1')
            c = conv_block(c, channels['c2'], CONV_FILTER_1711, relu, name='branch_c2')
            c = conv_block(c, channels['c3'], CONV_FILTER_7111, relu, name='branch_c3')
            c = conv_block(c, channels['c4'], CONV_FILTER_1711, relu, name='branch_c4')

            d = avg_pooling(x, CONV_FILTER_3311, name='branch_d0')
            d = conv_block(d, channels['d1'], CONV_FILTER_3311, relu, name='branch_d1')

            return concat([a, b, c, d], axis=3)

    @staticmethod
    def multi_pool_B(x, channels, name='multi_pool_B'):
        with tf.variable_scope(name):
            a = conv_block(x, channels['a0'], CONV_FILTER_1111, relu, name='branch_a0')
            a = conv_block(a, channels['a1'], CONV_FILTER_3322, relu, name='branch_a1')

            b = conv_block(x, channels['b0'], CONV_FILTER_1111, relu, name='branch_b0')
            b = conv_block(b, channels['b1'], CONV_FILTER_7111, relu, name='branch_b1')
            b = conv_block(b, channels['b2'], CONV_FILTER_1711, relu, name='branch_b2')
            b = conv_block(b, channels['b3'], CONV_FILTER_3322, relu, name='branch_b3')

            c = max_pooling(x, CONV_FILTER_3322, name='branch_c0')

            return concat([a, b, c], axis=3)

    @staticmethod
    def module_C(x, channels, name='module_C'):
        # type b?
        with tf.variable_scope(name):
            a = conv_block(x, channels['a0'], CONV_FILTER_1111, relu, name='branch_a0')

            b = conv_block(x, channels['b0'], CONV_FILTER_1111, relu, name='branch_b0')
            b0 = conv_block(b, channels['b1-0'], CONV_FILTER_3111, relu, name='branch_b1-0')
            b1 = conv_block(b, channels['b1-1'], CONV_FILTER_1311, relu, name='branch_b1-1')

            c = conv_block(x, channels['c0'], CONV_FILTER_1111, relu, name='branch_c0')
            c = conv_block(c, channels['c1'], CONV_FILTER_3311, relu, name='branch_c1')
            c0 = conv_block(c, channels['c2-0'], CONV_FILTER_1311, relu, name='branch_c2-0')
            c1 = conv_block(c, channels['c2-1'], CONV_FILTER_3111, relu, name='branch_c2-1')

            d = avg_pooling(x, CONV_FILTER_3311, name='branch_d0')
            d = conv_block(d, channels['d1'], CONV_FILTER_3311, relu, name='branch_d1')

            return concat([a, b0, b1, c0, c1, d], axis=3)

    def aux(self, x, n_classes, name='aux'):
        with tf.variable_scope(name):
            stack = Stacker(x)
            stack.avg_pooling(CONV_FILTER_5533)
            stack.conv_block(self.n_channel * 8, CONV_FILTER_1111, relu)

            filter_ = list(stack.last_layer.shape[1:3]) + [1, 1]
            stack.conv_block(self.n_channel * 48, filter_, relu)
            stack.flatten()
            logit = stack.linear(n_classes)
            proba = stack.softmax()
            return logit, proba

    def stem(self, stacker, name='stem'):
        with tf.variable_scope(name):
            stacker.add_layer(_residual_block, self.n_channel * 2, CONV_FILTER_3311, relu)
            stacker.add_layer(_residual_block, self.n_channel * 2, CONV_FILTER_3311, relu)
            stacker.add_layer(_residual_block, self.n_channel * 2, CONV_FILTER_3311, relu)
            self.skip_tensors += [stacker.last_layer]
            stacker.conv_block(self.n_channel * 2, CONV_FILTER_3322, relu)

            stacker.conv_block(self.n_channel * 4, CONV_FILTER_3311, relu)
            stacker.add_layer(_residual_block, self.n_channel * 4, CONV_FILTER_3311, relu)
            stacker.add_layer(_residual_block, self.n_channel * 4, CONV_FILTER_3311, relu)
            stacker.add_layer(_residual_block, self.n_channel * 4, CONV_FILTER_3311, relu)
            self.skip_tensors += [stacker.last_layer]
            stacker.max_pooling(CONV_FILTER_3322)

            stacker.conv_block(self.n_channel * 12, CONV_FILTER_3311, relu)
            stacker.add_layer(_residual_block, self.n_channel * 12, CONV_FILTER_3311, relu)
            stacker.add_layer(_residual_block, self.n_channel * 12, CONV_FILTER_3311, relu)
            stacker.add_layer(_residual_block, self.n_channel * 12, CONV_FILTER_3311, relu)
            self.skip_tensors += [stacker.last_layer]
            stacker.max_pooling(CONV_FILTER_3322)

            return stacker

    def build(self):
        with tf.variable_scope(self.name):
            self.skip_tensors = []
            self.stacker = Stacker(self.x)

            if self.x.shape[1:3] != self.resize_shape:
                self.stacker.resize_image(self.resize_shape)

            stacker = self.stem(self.stacker)

            a_channels0 = {
                'a0': self.n_channel * 4,
                'b0': self.n_channel * 3,
                'b1': self.n_channel * 4,
                'c0': self.n_channel * 4,
                'c1': self.n_channel * 6,
                'c2': self.n_channel * 6,
                'd1': self.n_channel * 2
            }
            a_channels1 = {
                'a0': self.n_channel * 4,
                'b0': self.n_channel * 3,
                'b1': self.n_channel * 4,
                'c0': self.n_channel * 4,
                'c1': self.n_channel * 6,
                'c2': self.n_channel * 6,
                'd1': self.n_channel * 4
            }
            a_channels2 = {
                'a0': self.n_channel * 4,
                'b0': self.n_channel * 3,
                'b1': self.n_channel * 4,
                'c0': self.n_channel * 4,
                'c1': self.n_channel * 6,
                'c2': self.n_channel * 6,
                'd1': self.n_channel * 4
            }

            stacker.add_layer(self.module_A, a_channels0)
            stacker = residual(stacker, self.module_A, a_channels0)
            stacker = residual(stacker, self.module_A, a_channels0)

            stacker.add_layer(self.module_A, a_channels1)
            stacker = residual(stacker, self.module_A, a_channels1)
            stacker = residual(stacker, self.module_A, a_channels1)

            stacker.add_layer(self.module_A, a_channels2)
            stacker = residual(stacker, self.module_A, a_channels2)
            stacker = residual(stacker, self.module_A, a_channels2)
            self.skip_tensors += [stacker.last_layer]

            a_b_multi_pool_channels = {
                'a0': self.n_channel * 24,

                'b0': self.n_channel * 4,
                'b1': self.n_channel * 6,
                'b2': self.n_channel * 6,

                'c0': self.n_channel * 4,
            }
            stacker.add_layer(self.multi_pool_A, a_b_multi_pool_channels)

            b_channels0 = {
                'a0': self.n_channel * 12,

                'b0': self.n_channel * 8,
                'b1': self.n_channel * 8,
                'b2': self.n_channel * 12,

                'c0': self.n_channel * 8,
                'c1': self.n_channel * 8,
                'c2': self.n_channel * 8,
                'c3': self.n_channel * 8,
                'c4': self.n_channel * 12,

                'd1': self.n_channel * 12,
            }
            b_channels1 = {
                'a0': self.n_channel * 12,

                'b0': self.n_channel * 10,
                'b1': self.n_channel * 10,
                'b2': self.n_channel * 12,

                'c0': self.n_channel * 10,
                'c1': self.n_channel * 10,
                'c2': self.n_channel * 10,
                'c3': self.n_channel * 10,
                'c4': self.n_channel * 12,

                'd1': self.n_channel * 12,
            }
            b_channels2 = {
                'a0': self.n_channel * 12,

                'b0': self.n_channel * 10,
                'b1': self.n_channel * 10,
                'b2': self.n_channel * 12,

                'c0': self.n_channel * 10,
                'c1': self.n_channel * 10,
                'c2': self.n_channel * 10,
                'c3': self.n_channel * 10,
                'c4': self.n_channel * 12,

                'd1': self.n_channel * 12,
            }
            b_channels3 = {
                'a0': self.n_channel * 12,

                'b0': self.n_channel * 12,
                'b1': self.n_channel * 12,
                'b2': self.n_channel * 12,

                'c0': self.n_channel * 12,
                'c1': self.n_channel * 12,
                'c2': self.n_channel * 12,
                'c3': self.n_channel * 12,
                'c4': self.n_channel * 12,

                'd1': self.n_channel * 12,
            }
            stacker.add_layer(self.module_B, b_channels0)
            stacker = residual(stacker, self.module_B, b_channels0)
            stacker = residual(stacker, self.module_B, b_channels0)

            stacker.add_layer(self.module_B, b_channels1)
            stacker = residual(stacker, self.module_B, b_channels1)
            stacker = residual(stacker, self.module_B, b_channels1)

            stacker.add_layer(self.module_B, b_channels2)
            stacker = residual(stacker, self.module_B, b_channels2)
            stacker = residual(stacker, self.module_B, b_channels2)

            stacker.add_layer(self.module_B, b_channels3)
            stacker = residual(stacker, self.module_B, b_channels3)
            stacker = residual(stacker, self.module_B, b_channels3)
            # self.aux_logit, self.aux_proba = self.aux(stacker.last_layer, self.n_classes)
            self.skip_tensors += [stacker.last_layer]

            b_c_multi_pool_channels = {
                'a0': self.n_channel * 12,
                'a1': self.n_channel * 20,

                'b0': self.n_channel * 12,
                'b1': self.n_channel * 12,
                'b2': self.n_channel * 12,
                'b3': self.n_channel * 12,
            }
            stacker.add_layer(self.multi_pool_B, b_c_multi_pool_channels)

            c_channels0 = {
                'a0': self.n_channel * 20,

                'b0': self.n_channel * 24,
                'b1-0': self.n_channel * 24,
                'b1-1': self.n_channel * 24,

                'c0': self.n_channel * 28,
                'c1': self.n_channel * 24,
                'c2-0': self.n_channel * 24,
                'c2-1': self.n_channel * 24,

                'd1': self.n_channel * 12,
            }
            c_channels1 = {
                'a0': self.n_channel * 20,

                'b0': self.n_channel * 24,
                'b1-0': self.n_channel * 24,
                'b1-1': self.n_channel * 24,

                'c0': self.n_channel * 28,
                'c1': self.n_channel * 24,
                'c2-0': self.n_channel * 24,
                'c2-1': self.n_channel * 24,

                'd1': self.n_channel * 12,
            }
            stacker.add_layer(self.module_C, c_channels0)
            stacker = residual(stacker, self.module_C, c_channels0)
            stacker = residual(stacker, self.module_C, c_channels0)
            stacker.add_layer(self.module_C, c_channels1)
            stacker = residual(stacker, self.module_C, c_channels1)
            stacker = residual(stacker, self.module_C, c_channels1)
            self.last_layer = stacker.last_layer
            self.skip_tensors += [stacker.last_layer]


class InceptionV2UnetDecoderModule(BaseInceptionNetModule):

    def __init__(self, x, n_classes, skip_tensors, capacity=None, reuse=False, resize_shape=(299, 299), name=None,
                 verbose=0):
        super().__init__(x, n_classes, capacity, reuse, resize_shape, name, verbose)
        self.skip_tensors = skip_tensors

    @staticmethod
    def module_A(x, channels, name='module_A'):
        with tf.variable_scope(name):
            a = conv_block(x, channels['a0'], CONV_FILTER_1111, relu, name='branch_a0')

            b = conv_block(x, channels['b0'], CONV_FILTER_1111, relu, name='branch_b0')
            b = conv_block(b, channels['b1'], CONV_FILTER_5511, relu, name='branch_b1')

            c = conv_block(x, channels['c0'], CONV_FILTER_1111, relu, name='branch_c0')
            c = conv_block(c, channels['c1'], CONV_FILTER_3311, relu, name='branch_c1')
            c = conv_block(c, channels['c2'], CONV_FILTER_3311, relu, name='branch_c2')

            d = avg_pooling(x, CONV_FILTER_3311, name='branch_d0')
            d = conv_block(d, channels['d1'], CONV_FILTER_1111, relu, name='branch_d1')

            return concat([a, b, c, d], axis=3)

    @staticmethod
    def multi_conv_transpose_A(x, channels, name='multi_conv_transpose_A'):
        with tf.variable_scope(name):
            a = upscale_2x_block(x, channels['a0'], CONV_FILTER_3322, relu, name='branch_a0')

            b = conv_block(x, channels['b0'], CONV_FILTER_1111, relu, name='branch_b0')
            b = conv_block(b, channels['b1'], CONV_FILTER_3311, relu, name='branch_b1')
            b = upscale_2x_block(b, channels['b2'], CONV_FILTER_3322, relu, name='branch_b2')

            # # c = max_pooling(x, CONV_FILTER_3322, name='branch_c0')
            # c = upscale_2x_block(x, x.shape[-1], CONV_FILTER_1122, relu, name='branch_c0')

            # return concat([a, b, c], axis=3)
            return concat([a, b], axis=3)

    @staticmethod
    def module_B(x, channels, name='module_B'):
        # mixed4: 17 x 17 x 768.

        with tf.variable_scope(name):
            a = conv_block(x, channels['a0'], CONV_FILTER_1111, relu, name='branch_a0')

            b = conv_block(x, channels['b0'], CONV_FILTER_1111, relu, name='branch_b0')
            b = conv_block(b, channels['b1'], CONV_FILTER_7111, relu, name='branch_b1')
            b = conv_block(b, channels['b2'], CONV_FILTER_1711, relu, name='branch_b2')

            c = conv_block(x, channels['c0'], CONV_FILTER_1111, relu, name='branch_c0')
            c = conv_block(c, channels['c1'], CONV_FILTER_7111, relu, name='branch_c1')
            c = conv_block(c, channels['c2'], CONV_FILTER_1711, relu, name='branch_c2')
            c = conv_block(c, channels['c3'], CONV_FILTER_7111, relu, name='branch_c3')
            c = conv_block(c, channels['c4'], CONV_FILTER_1711, relu, name='branch_c4')

            d = avg_pooling(x, CONV_FILTER_3311, name='branch_d0')
            d = conv_block(d, channels['d1'], CONV_FILTER_3311, relu, name='branch_d1')

            return concat([a, b, c, d], axis=3)

    @staticmethod
    def multi_conv_transpose_B(x, channels, name='multi_pool_B'):
        with tf.variable_scope(name):
            a = conv_block(x, channels['a0'], CONV_FILTER_1111, relu, name='branch_a0')
            a = upscale_2x_block(a, channels['a1'], CONV_FILTER_3322, relu, name='branch_a1')

            b = conv_block(x, channels['b0'], CONV_FILTER_1111, relu, name='branch_b0')
            b = conv_block(b, channels['b1'], CONV_FILTER_7111, relu, name='branch_b1')
            b = conv_block(b, channels['b2'], CONV_FILTER_1711, relu, name='branch_b2')
            b = upscale_2x_block(b, channels['b3'], CONV_FILTER_3322, relu, name='branch_b3')

            # c = max_pooling(x, CONV_FILTER_3322, name='branch_c0')
            # c = upscale_2x_block(x, x.shape[-1], CONV_FILTER_1122, relu, name='branch_c0')

            # return concat([a, b, c], axis=3)
            return concat([a, b], axis=3)

    @staticmethod
    def module_C(x, channels, name='module_C'):
        # type b?
        with tf.variable_scope(name):
            a = conv_block(x, channels['a0'], CONV_FILTER_1111, relu, name='branch_a0')

            b = conv_block(x, channels['b0'], CONV_FILTER_1111, relu, name='branch_b0')
            b0 = conv_block(b, channels['b1-0'], CONV_FILTER_3111, relu, name='branch_b1-0')
            b1 = conv_block(b, channels['b1-1'], CONV_FILTER_1311, relu, name='branch_b1-1')

            c = conv_block(x, channels['c0'], CONV_FILTER_1111, relu, name='branch_c0')
            c = conv_block(c, channels['c1'], CONV_FILTER_3311, relu, name='branch_c1')
            c0 = conv_block(c, channels['c2-0'], CONV_FILTER_1311, relu, name='branch_c2-0')
            c1 = conv_block(c, channels['c2-1'], CONV_FILTER_3111, relu, name='branch_c2-1')

            d = avg_pooling(x, CONV_FILTER_3311, name='branch_d0')
            d = conv_block(d, channels['d1'], CONV_FILTER_3311, relu, name='branch_d1')

            return concat([a, b0, b1, c0, c1, d], axis=3)

    def aux(self, x, n_classes, name='aux'):
        with tf.variable_scope(name):
            stack = Stacker(x)
            stack.avg_pooling(CONV_FILTER_5533)
            stack.conv_block(self.n_channel * 8, CONV_FILTER_1111, relu)

            filter_ = list(stack.last_layer.shape[1:3]) + [1, 1]
            stack.conv_block(self.n_channel * 48, filter_, relu)
            stack.flatten()
            logit = stack.linear(n_classes)
            proba = stack.softmax()
            return logit, proba

    def stem_transpose(self, stacker, name='stem_transpose'):
        skip_tensors = self.skip_tensors[::-1]

        with tf.variable_scope(name):
            stacker.conv_block(self.n_channel * 12, CONV_FILTER_3311, relu)
            stacker.add_layer(_residual_block, self.n_channel * 12, CONV_FILTER_3311, relu)
            stacker.add_layer(_residual_block, self.n_channel * 12, CONV_FILTER_3311, relu)
            stacker.add_layer(_residual_block, self.n_channel * 12, CONV_FILTER_3311, relu)
            stacker.upscale_2x_block(self.n_channel * 12, CONV_FILTER_3322, relu)
            stacker.concat(skip_tensors[3], axis=3)

            stacker.conv_block(self.n_channel * 12, CONV_FILTER_3311, relu)
            stacker.add_layer(_residual_block, self.n_channel * 12, CONV_FILTER_3311, relu)
            stacker.add_layer(_residual_block, self.n_channel * 12, CONV_FILTER_3311, relu)
            stacker.add_layer(_residual_block, self.n_channel * 12, CONV_FILTER_3311, relu)
            stacker.conv_block(self.n_channel * 5, CONV_FILTER_1111, relu)
            stacker.upscale_2x_block(self.n_channel * 4, CONV_FILTER_3322, relu)
            stacker.concat(skip_tensors[4], axis=3)

            stacker.conv_block(self.n_channel * 4, CONV_FILTER_3311, relu)
            stacker.add_layer(_residual_block, self.n_channel * 4, CONV_FILTER_3311, relu)
            stacker.add_layer(_residual_block, self.n_channel * 4, CONV_FILTER_3311, relu)
            stacker.add_layer(_residual_block, self.n_channel * 4, CONV_FILTER_3311, relu)
            stacker.upscale_2x_block(self.n_channel * 2, CONV_FILTER_3322, relu)
            stacker.concat(skip_tensors[5], axis=3)

            stacker.conv_block(self.n_channel * 2, CONV_FILTER_3311, relu)
            stacker.add_layer(_residual_block, self.n_channel * 2, CONV_FILTER_3311, relu)
            stacker.add_layer(_residual_block, self.n_channel * 2, CONV_FILTER_3311, relu)
            stacker.add_layer(_residual_block, self.n_channel * 2, CONV_FILTER_3311, relu)

            return stacker

    def build(self):
        with tf.variable_scope(self.name):
            # x expect image NHWC format
            skip_tensors = self.skip_tensors[::-1]

            stacker = Stacker(self.x)
            stacker.concat(skip_tensors[0], axis=3)

            c_channels1 = {
                'a0': self.n_channel * 20,

                'b0': self.n_channel * 24,
                'b1-0': self.n_channel * 24,
                'b1-1': self.n_channel * 24,

                'c0': self.n_channel * 28,
                'c1': self.n_channel * 24,
                'c2-0': self.n_channel * 24,
                'c2-1': self.n_channel * 24,

                'd1': self.n_channel * 12,
            }
            stacker.add_layer(self.module_C, c_channels1)
            stacker = residual(stacker, self.module_C, c_channels1)
            stacker = residual(stacker, self.module_C, c_channels1)

            c_channels0 = {
                'a0': self.n_channel * 20,

                'b0': self.n_channel * 24,
                'b1-0': self.n_channel * 24,
                'b1-1': self.n_channel * 24,

                'c0': self.n_channel * 28,
                'c1': self.n_channel * 24,
                'c2-0': self.n_channel * 24,
                'c2-1': self.n_channel * 24,

                'd1': self.n_channel * 12,
            }
            stacker.add_layer(self.module_C, c_channels0)
            stacker = residual(stacker, self.module_C, c_channels0)
            stacker = residual(stacker, self.module_C, c_channels0)

            b_c_multi_pool_channels = {
                'a0': self.n_channel * 12,
                'a1': self.n_channel * 20,
                'b0': self.n_channel * 12,
                'b1': self.n_channel * 12,
                'b2': self.n_channel * 12,
                'b3': self.n_channel * 12,
            }
            stacker.add_layer(self.multi_conv_transpose_B, b_c_multi_pool_channels)

            # self.aux_logit, self.aux_proba = self.aux(stacker.last_layer, self.n_classes)
            stacker.concat(skip_tensors[1], axis=3)
            b_channels3 = {
                'a0': self.n_channel * 12,

                'b0': self.n_channel * 12,
                'b1': self.n_channel * 12,
                'b2': self.n_channel * 12,

                'c0': self.n_channel * 12,
                'c1': self.n_channel * 12,
                'c2': self.n_channel * 12,
                'c3': self.n_channel * 12,
                'c4': self.n_channel * 12,

                'd1': self.n_channel * 12,
            }
            stacker.add_layer(self.module_B, b_channels3)
            stacker = residual(stacker, self.module_B, b_channels3)
            stacker = residual(stacker, self.module_B, b_channels3)
            b_channels2 = {
                'a0': self.n_channel * 12,

                'b0': self.n_channel * 10,
                'b1': self.n_channel * 10,
                'b2': self.n_channel * 12,

                'c0': self.n_channel * 10,
                'c1': self.n_channel * 10,
                'c2': self.n_channel * 10,
                'c3': self.n_channel * 10,
                'c4': self.n_channel * 12,

                'd1': self.n_channel * 12,
            }
            stacker.add_layer(self.module_B, b_channels2)
            stacker = residual(stacker, self.module_B, b_channels2)
            stacker = residual(stacker, self.module_B, b_channels2)
            b_channels1 = {
                'a0': self.n_channel * 12,

                'b0': self.n_channel * 10,
                'b1': self.n_channel * 10,
                'b2': self.n_channel * 12,

                'c0': self.n_channel * 10,
                'c1': self.n_channel * 10,
                'c2': self.n_channel * 10,
                'c3': self.n_channel * 10,
                'c4': self.n_channel * 12,

                'd1': self.n_channel * 12,
            }
            stacker.add_layer(self.module_B, b_channels1)
            stacker = residual(stacker, self.module_B, b_channels1)
            stacker = residual(stacker, self.module_B, b_channels1)
            b_channels0 = {
                'a0': self.n_channel * 12,

                'b0': self.n_channel * 8,
                'b1': self.n_channel * 8,
                'b2': self.n_channel * 12,

                'c0': self.n_channel * 8,
                'c1': self.n_channel * 8,
                'c2': self.n_channel * 8,
                'c3': self.n_channel * 8,
                'c4': self.n_channel * 12,

                'd1': self.n_channel * 12,
            }
            stacker.add_layer(self.module_B, b_channels0)
            stacker = residual(stacker, self.module_B, b_channels0)
            stacker = residual(stacker, self.module_B, b_channels0)
            a_b_multi_pool_channels = {
                'a0': self.n_channel * 24,

                'b0': self.n_channel * 4,
                'b1': self.n_channel * 6,
                'b2': self.n_channel * 6,

                'c0': self.n_channel * 4,
            }
            stacker.add_layer(self.multi_conv_transpose_A, a_b_multi_pool_channels)

            stacker.concat(skip_tensors[2], axis=3)
            a_channels2 = {
                'a0': self.n_channel * 4,
                'b0': self.n_channel * 3,
                'b1': self.n_channel * 4,
                'c0': self.n_channel * 4,
                'c1': self.n_channel * 6,
                'c2': self.n_channel * 6,
                'd1': self.n_channel * 4
            }
            stacker.add_layer(self.module_A, a_channels2)
            stacker = residual(stacker, self.module_A, a_channels2)
            stacker = residual(stacker, self.module_A, a_channels2)
            a_channels1 = {
                'a0': self.n_channel * 4,
                'b0': self.n_channel * 3,
                'b1': self.n_channel * 4,
                'c0': self.n_channel * 4,
                'c1': self.n_channel * 6,
                'c2': self.n_channel * 6,
                'd1': self.n_channel * 4
            }
            stacker.add_layer(self.module_A, a_channels1)
            stacker = residual(stacker, self.module_A, a_channels1)
            stacker = residual(stacker, self.module_A, a_channels1)

            a_channels0 = {
                'a0': self.n_channel * 4,
                'b0': self.n_channel * 3,
                'b1': self.n_channel * 4,
                'c0': self.n_channel * 4,
                'c1': self.n_channel * 6,
                'c2': self.n_channel * 6,
                'd1': self.n_channel * 2
            }
            stacker.add_layer(self.module_A, a_channels0)
            stacker = residual(stacker, self.module_A, a_channels0)
            stacker = residual(stacker, self.module_A, a_channels0)

            stacker = self.stem_transpose(stacker)
            stacker.resize_image(self.resize_shape)

            self.decode = stacker.last_layer


def _residual_block(x, n_channel, filter_, activation, name='residual_block'):
    with tf.variable_scope(name):
        x_in = x
        x = conv_block(x, n_channel, filter_, activation, name='conv_block1')
        x = conv_block(x, n_channel, filter_, activation, name='conv_block2')
        x = conv_block(x, n_channel, filter_, activation, name='conv_block3')
        x = residual_add(x, x_in)
    return x


class InceptionUNetModule(BaseNetModule):
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

    def bottom_layer(self, x, n_channel, depth, dropout_rate_tensor):
        stacker = Stacker(x, name='bottom')
        stacker = self.conv_seq(stacker, n_channel, depth, dropout_rate_tensor)
        self.bottom_layer_stacker = stacker

        return self.bottom_layer_stacker.last_layer

    def build(self):

        self.DynamicDropoutRate = DynamicDropoutRate(self.dropout_rate)
        self.drop_out_tensor = self.DynamicDropoutRate.tensor

        with tf.variable_scope(self.name, reuse=self.reuse):
            self.encoder = InceptionV2UNetEncoderModule(
                self.x, None, resize_shape=(201, 201),
                capacity=self.capacity
            )
            self.encoder.build()
            encode = self.encoder.last_layer
            skip_tensors = self.encoder.skip_tensors

            bottom_layer = self.bottom_layer(
                encode,
                self.capacity * 128,
                self.depth,
                self.drop_out_tensor
            )

            self.decoder = InceptionV2UnetDecoderModule(
                bottom_layer, None, skip_tensors,
                capacity=self.capacity,
                resize_shape=(101, 101),
            )
            self.decoder.build()
            decode = self.decoder.decode

            stacker = Stacker(decode, name='to_match')
            stacker.conv_block(self.n_classes, CONV_FILTER_3311, relu)
            self.logit = stacker.last_layer
            self.proba = pixel_wise_softmax(self.logit)

    def set_train(self, sess):
        self.DynamicDropoutRate.set_train(sess)

    def set_predict(self, sess):
        self.DynamicDropoutRate.set_predict(sess)

    def update_dropout_rate(self, sess, x):
        self.dropout_rate = x
        self.DynamicDropoutRate.update(sess, x)
