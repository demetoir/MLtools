from script.model.sklearn_like_model.net_structure.Base_net_structure import Base_net_structure
from script.util.Stacker import Stacker
from script.util.tensor_ops import *


def inception_module_naive(x, n_channel, name='inception_module_naive'):
    with tf.variable_scope(name):
        a = conv_block(x, n_channel, CONV_FILTER_1111, relu, name='branch_a')
        b = conv_block(x, n_channel, CONV_FILTER_3311, relu, name='branch_b')
        c = conv_block(x, n_channel, CONV_FILTER_5511, relu, name='branch_c')
        d = max_pooling(x, CONV_FILTER_3311, name='branch_d')
        return concat([a, b, c, d], axis=3)


def inception_module_dimension_reduction(x, channels, name='inception_module_demention_reduction'):
    channels = {
        'a0': 64,
        'b0': 96,
        'b1': 128,
        'c0': 16,
        'c1': 32,
        'd1': 64,
    }

    with tf.variable_scope(name):
        a = conv_block(x, channels['a0'], CONV_FILTER_1111, relu, name='branch_a0')

        b = conv_block(x, channels['b0'], CONV_FILTER_1111, relu, name='branch_b0')
        b = conv_block(b, channels['b1'], CONV_FILTER_3311, relu, name='branch_b1')

        c = conv_block(x, channels['c0'], CONV_FILTER_1111, relu, name='branch_c0')
        c = conv_block(c, channels['c1'], CONV_FILTER_5511, relu, name='branch_c1')

        d = max_pooling(x, CONV_FILTER_3311, name='branch_d0')
        d = conv_block(d, channels['d1'], CONV_FILTER_1111, relu, name='branch_d1')

        return concat([a, b, c, d], axis=3)


def inception_module_factorizing(x, channels, name='inception_module_factorizing'):
    channels = {
        'a0': 64,
        'b0': 96,
        'b1': 128,
        'c0': 16,
        'c1': 32,
        'c2': 64,
        'd1': 64,
    }

    with tf.variable_scope(name):
        a = conv_block(x, channels['a0'], CONV_FILTER_1111, relu, name='branch_a0')

        b = conv_block(x, channels['b0'], CONV_FILTER_1111, relu, name='branch_b0')
        b = conv_block(b, channels['b1'], CONV_FILTER_3311, relu, name='branch_b1')

        c = conv_block(x, channels['c0'], CONV_FILTER_1111, relu, name='branch_c0')
        c = conv_block(c, channels['c1'], CONV_FILTER_3311, relu, name='branch_c1')
        c = conv_block(c, channels['c2'], CONV_FILTER_3311, relu, name='branch_c2')

        d = max_pooling(x, CONV_FILTER_3311, name='branch_d1')
        d = conv_block(d, channels['d1'], CONV_FILTER_1111, relu, name='branch_d1')

        return concat([a, b, c, d], axis=3)


def inception_module_factorizing_asymmetric(x, n_size, name='inception_module_factorizing_asymmetric'):
    channels = {
        'a0': 64,
        'b0': 96,
        'b1': 128,
        'c0': 16,
        'c1': 32,
        'c2': 64,
        'd1': 64,
    }

    with tf.variable_scope(name):
        a = conv_block(x, channels['a0'], CONV_FILTER_1111, relu, name='branch_a0')

        b = conv_block(x, channels['b0'], CONV_FILTER_1111, relu, name='branch_b0')
        # n*1 1*n
        b = conv_block(b, channels['b1'], CONV_FILTER_3311, relu, name='branch_b1')
        b = conv_block(b, channels['b2'], CONV_FILTER_3311, relu, name='branch_b2')

        c = conv_block(x, channels['c0'], CONV_FILTER_1111, relu, name='branch_c0')
        # n*1 1*n
        c = conv_block(c, channels['c1'], CONV_FILTER_3311, relu, name='branch_c1')
        c = conv_block(c, channels['c2'], CONV_FILTER_3311, relu, name='branch_c2')
        # n*1 1*n
        c = conv_block(c, channels['c3'], CONV_FILTER_3311, relu, name='branch_c3')
        c = conv_block(c, channels['c4'], CONV_FILTER_3311, relu, name='branch_c4')

        d = max_pooling(x, CONV_FILTER_3311, name='branch_d0')
        d = conv_block(d, channels['d1'], CONV_FILTER_1111, relu, name='branch_d1')

        return concat([a, b, c, d], axis=3)


def inception_module_v2_c(x, name='inception_module_v2_c'):
    channels = {
        'a0': 64,
        'b0': 96,
        'b1': 128,
        'b2': 128,
        'c0': 16,
        'c1': 32,
        'c2': 64,
        'c3': 64,
        'c4': 64,
        'd1': 64,
    }
    with tf.variable_scope(name):
        a = conv_block(x, channels['a0'], CONV_FILTER_1111, relu, name='branch_a0')

        b = conv_block(x, channels['b0'], CONV_FILTER_1111, relu, name='branch_b0')
        # n*1 1*n
        b1 = conv_block(b, channels['b1'], CONV_FILTER_3311, relu, name='branch_b1')
        b2 = conv_block(b, channels['b2'], CONV_FILTER_3311, relu, name='branch_b2')

        c = conv_block(x, channels['c0'], CONV_FILTER_1111, relu, name='branch_c0')
        c = conv_block(c, channels['c1'], CONV_FILTER_3311, relu, name='branch_c1')
        # n*1 1*n
        c1 = conv_block(c, channels['c3'], CONV_FILTER_3311, relu, name='branch_c3')
        c2 = conv_block(c, channels['c4'], CONV_FILTER_3311, relu, name='branch_c4')

        d = max_pooling(x, CONV_FILTER_3311, name='branch_d0')
        d = conv_block(d, channels['d1'], CONV_FILTER_1111, relu, name='branch_d1')

        return concat([a, b1, b2, c1, c2, d], axis=3)


def inception_module_max_pool_half(x, channels, name='inception_module_max_pool_half'):
    channels = {
        'a0': 0,
        'a1': 0,
        'b0': 0,
        'b1': 0,
        'b2': 0,
    }
    with tf.variable_scope(name):
        a = conv_block(x, channels['a0'], CONV_FILTER_1111, relu)
        a = conv_block(a, channels['a0'], CONV_FILTER_3322, relu)

        b = conv_block(x, channels['b0'], CONV_FILTER_1111, relu)
        b = conv_block(b, channels['b1'], CONV_FILTER_3311, relu)
        b = conv_block(b, channels['b2'], CONV_FILTER_3322, relu)

        c = max_pooling(x, CONV_FILTER_2222)
        return concat([a, b, c], axis=3)


def inception_v4_stem():
    pass


def inception_v4_A():
    pass


def inception_v4_B():
    pass


def inception_v4_C():
    pass


def inception_v4_reduction_A():
    pass


def inception_v4_reduction_B():
    pass


def inception_resnet_v1_stem():
    pass


def inception_resnet_V2_stem():
    pass


class InceptionStructure(Base_net_structure):

    def __init__(self, x, n_classes, model_type=3, reuse=False, name=None, verbose=0):
        super().__init__(reuse, name, verbose)
        self.x = x
        self.n_classes = n_classes
        self.model_type = model_type
        self.model_func = {
            1: self.v1,
            2: self.v2,
            3: self.v3,
            4: self.v4,
        }

    def stem_v1(self, stacker, n_channel, name='stem_v1'):
        with tf.variable_scope(name):
            stacker.conv_block(64, CONV_FILTER_7722, relu)
            stacker.max_pooling(CONV_FILTER_3322)
            # LRN
            # tf.nn.local_response_normalization()
            stacker.bn()
            stacker.conv_block(64, CONV_FILTER_1111, relu)
            stacker.conv_block(192, CONV_FILTER_3311, relu)
            # LRN
            # tf.nn.local_response_normalization()
            # alpha = 0.0001, k = 1, beta = 0.75, n = 5,

            stacker.bn()
            stacker.max_pooling(CONV_FILTER_3322)

            return stacker

    def stem_v2(self, stacker, n_channel, name='stem_v2'):
        with tf.variable_scope(name):
            stacker.conv_block(32, CONV_FILTER_3322, relu)
            stacker.conv_block(32, CONV_FILTER_3311, relu)
            # padded?
            stacker.conv_block(64, CONV_FILTER_3311, relu)
            stacker.max_pooling(CONV_FILTER_3322)

            stacker.conv_block(64, CONV_FILTER_3311, relu)
            stacker.conv_block(80, CONV_FILTER_3322, relu)
            stacker.conv_block(192, CONV_FILTER_3311, relu)

            # LRN
            # tf.nn.local_response_normalization()
            # alpha = 0.0001, k = 1, beta = 0.75, n = 5,

            stacker.bn()
            stacker.max_pooling(CONV_FILTER_3322)

            return stacker

    def aux(self, x, n_channel, name='aux'):
        with tf.variable_scope(name):
            stacker = Stacker(x)
            stacker.avg_pooling(CONV_FILTER_5533)
            stacker.conv_block(n_channel, CONV_FILTER_1111, relu)
            stacker.flatten()
            stacker.linear_block(n_channel, relu)
            stacker.linear_block(n_channel, relu)
            stacker.linear_block(self.n_classes, relu)
            logit = stacker.last_layer
            stacker.softmax()
            proba = stacker.last_layer

            return logit, proba

    def build(self):
        with tf.variable_scope(self.name):
            self.stacker = Stacker(self.x)

            self.logit, self.proba = self.v1(self.stacker)

    def v1(self, stacker):
        channels_3a = {
            'a0': 64,
            'b0': 96,
            'b1': 128,
            'c0': 16,
            'c1': 32,
            'd1': 32,
        }
        channels_3b = {
            'a0': 128,
            'b0': 128,
            'b1': 196,
            'c0': 32,
            'c1': 96,
            'd1': 64,
        }
        channels_4a = {
            'a0': 192,
            'b0': 96,
            'b1': 208,
            'c0': 16,
            'c1': 48,
            'd1': 64,
        }
        channels_4b = {
            'a0': 160,
            'b0': 112,
            'b1': 224,
            'c0': 24,
            'c1': 64,
            'd1': 64,
        }
        channels_4c = {
            'a0': 128,
            'b0': 128,
            'b1': 256,
            'c0': 24,
            'c1': 64,
            'd1': 64,
        }
        channels_4d = {
            'a0': 112,
            'b0': 144,
            'b1': 288,
            'c0': 32,
            'c1': 64,
            'd1': 64,
        }
        channels_4e = {
            'a0': 256,
            'b0': 160,
            'b1': 320,
            'c0': 32,
            'c1': 128,
            'd1': 128,
        }
        channels_5a = {
            'a0': 256,
            'b0': 160,
            'b1': 320,
            'c0': 32,
            'c1': 128,
            'd1': 128,
        }
        channels_5b = {
            'a0': 384,
            'b0': 192,
            'b1': 384,
            'c0': 48,
            'c1': 128,
            'd1': 128,
        }

        self.stacker.resize_image((224, 224))
        stacker = self.stem_v1(stacker, 64)

        stacker.add_layer(inception_module_dimension_reduction, channels_3a)
        stacker.add_layer(inception_module_dimension_reduction, channels_3b)
        stacker.max_pooling(CONV_FILTER_3322)

        stacker.add_layer(inception_module_dimension_reduction, channels_4a)
        aux0_logit, aux0_proba = self.aux(stacker.last_layer, 64, 'aux0')
        stacker.add_layer(inception_module_dimension_reduction, channels_4b)
        stacker.add_layer(inception_module_dimension_reduction, channels_4c)
        stacker.add_layer(inception_module_dimension_reduction, channels_4d)
        aux1_logit, aux1_proba = self.aux(stacker.last_layer, 64, 'aux1')
        stacker.add_layer(inception_module_dimension_reduction, channels_4e)
        stacker.max_pooling(CONV_FILTER_3322)

        stacker.add_layer(inception_module_dimension_reduction, channels_5a)
        stacker.add_layer(inception_module_dimension_reduction, channels_5b)
        stacker.max_pooling(CONV_FILTER_7777)

        # dropout
        stacker.flatten()
        stacker.linear_block(1000, relu)
        stacker.linear_block(self.n_classes)
        logit = stacker.last_layer
        stacker.softmax()
        proba = stacker.last_layer

        return logit, proba

    def v2(self, stacker):
        self.stacker.resize_image((299, 299))

        # inception refactorize * 3
        # inception refactorize  asymetric 6 * 5
        # inception 7 * 2

        stacker.max_pooling((8, 8, 8, 8))
        # dropout
        stacker.flatten()
        stacker.linear_block(1000, relu)
        stacker.linear_block(self.n_classes)
        logit = stacker.last_layer
        stacker.softmax()
        proba = stacker.last_layer
        return logit, proba

    def v3(self):
        pass

    def v4(self, stacker):
        self.stacker.resize_image((299, 299))
        # stem
        # A * 4
        # reduction A
        # b * 7
        # reduction B
        # C * 3
        # avg pool

        # dropout
        # softmax

        pass
