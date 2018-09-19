from script.model.sklearn_like_model.NetModule.InceptionSructure.BaseInceptionNetModule import \
    BaseInceptionNetModule
from script.util.Stacker import Stacker
from script.util.tensor_ops import *


class InceptionV2NetModule(BaseInceptionNetModule):
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
    def multipool_B(x, channels, name='multipool_B'):
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
        Stacker()
        # stacker = Stacker(x)
        with tf.variable_scope(name):
            stacker.conv_block(self.n_channel * 2, CONV_FILTER_3322, relu)
            stacker.conv_block(self.n_channel * 2, CONV_FILTER_3311, relu)

            stacker.conv_block(self.n_channel * 4, CONV_FILTER_3311, relu)
            stacker.max_pooling(CONV_FILTER_3322)

            stacker.conv_block(self.n_channel * 5, CONV_FILTER_1111, relu)
            stacker.conv_block(self.n_channel * 12, CONV_FILTER_3311, relu)
            stacker.max_pooling(CONV_FILTER_3322)

            return stacker

    def structure(self, stacker):
        stacker.resize_image((299, 299))

        stacker = self.stem(stacker)

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
        stacker.add_layer(self.module_A, a_channels1)
        stacker.add_layer(self.module_A, a_channels2)

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
        stacker.add_layer(self.module_B, b_channels1)
        stacker.add_layer(self.module_B, b_channels2)
        stacker.add_layer(self.module_B, b_channels3)
        self.aux_logit, self.aux_proba = self.aux(stacker.last_layer, self.n_classes)

        b_c_multipool_channels = {
            'a0': self.n_channel * 12,
            'a1': self.n_channel * 20,

            'b0': self.n_channel * 12,
            'b1': self.n_channel * 12,
            'b2': self.n_channel * 12,
            'b3': self.n_channel * 12,
        }
        stacker.add_layer(self.multipool_B, b_c_multipool_channels)

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
        stacker.add_layer(self.module_C, c_channels1)

        # inception refactorize * 3
        # inception refactorize asymetric 6 * 5
        # inception 7 * 2

        stacker.max_pooling((8, 8, 8, 8))
        # dropout
        stacker.flatten()
        stacker.linear_block(self.n_channel * 64, relu)
        stacker.linear(self.n_classes)
        logit = stacker.last_layer
        stacker.softmax()
        proba = stacker.last_layer
        return logit, proba

    def build(self):
        with tf.variable_scope(self.name):
            self.stacker = Stacker(self.x)

            self.logit, self.proba = self.structure(self.stacker)
