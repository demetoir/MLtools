from script.model.sklearn_like_model.net_structure.Base_net_structure import Base_net_structure
from script.util.Stacker import Stacker
from script.util.tensor_ops import *


def inception_v1_dimension_reduction(x, channels, name='inception_v1_dimension_reduction'):
    with tf.variable_scope(name):
        a = conv_block(x, channels['a0'], CONV_FILTER_1111, relu, name='branch_a0')

        b = conv_block(x, channels['b0'], CONV_FILTER_1111, relu, name='branch_b0')
        b = conv_block(b, channels['b1'], CONV_FILTER_3311, relu, name='branch_b1')

        c = conv_block(x, channels['c0'], CONV_FILTER_1111, relu, name='branch_c0')
        c = conv_block(c, channels['c1'], CONV_FILTER_5511, relu, name='branch_c1')

        d = max_pooling(x, CONV_FILTER_3311, name='branch_d0')
        d = conv_block(d, channels['d1'], CONV_FILTER_1111, relu, name='branch_d1')

        return concat([a, b, c, d], axis=3)


def inception_v1_factorizing(x, channels, name='inception_v1_factorizing'):
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


def inception_v1_factorizing_asymmetric(x, n_size, name='inception_v1_factorizing_asymmetric'):
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


def inception_v2_module_a(x, channels, name='inception_v2_module_a'):
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


def inception_v2_module_a_b_multi_pool(x, channels, name='inception_v2_module_a_b_multi_pool'):
    with tf.variable_scope(name):
        a = conv_block(x, channels['a0'], CONV_FILTER_3322, relu, name='branch_a0')

        b = conv_block(x, channels['b0'], CONV_FILTER_1111, relu, name='branch_b0')
        b = conv_block(b, channels['b1'], CONV_FILTER_3311, relu, name='branch_b1')
        b = conv_block(b, channels['b2'], CONV_FILTER_3322, relu, name='branch_b2')

        c = max_pooling(x, CONV_FILTER_3322, name='branch_c0')

        return concat([a, b, c], axis=3)


def inception_v2_module_b(x, channels, name='inception_v2_module_b'):
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


def inception_v2_module_b_c_multipool(x, channels, name='inception_v2_module_b_c_multipool'):
    with tf.variable_scope(name):
        a = conv_block(x, channels['a0'], CONV_FILTER_1111, relu, name='branch_a0')
        a = conv_block(a, channels['a1'], CONV_FILTER_3322, relu, name='branch_a1')

        b = conv_block(x, channels['b0'], CONV_FILTER_1111, relu, name='branch_b0')
        b = conv_block(b, channels['b1'], CONV_FILTER_7111, relu, name='branch_b1')
        b = conv_block(b, channels['b2'], CONV_FILTER_1711, relu, name='branch_b2')
        b = conv_block(b, channels['b3'], CONV_FILTER_3322, relu, name='branch_b3')

        c = max_pooling(x, CONV_FILTER_3322, name='branch_c0')

        return concat([a, b, c], axis=3)


def inception_v2_module_c(x, channels, name='inception_v2_module_c'):
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


def inception_v2_aux(x, n_classes, name='inception_v2_aux'):
    with tf.variable_scope(name):
        stack = Stacker(x)
        stack.avg_pooling(CONV_FILTER_5533)
        stack.conv_block(128, CONV_FILTER_1111, relu)

        filter_ = list(stack.last_layer.shape[1:3]) + [1, 1]
        stack.conv_block(768, filter_, relu)
        stack.flatten()
        logit = stack.linear(n_classes)
        proba = stack.softmax()
        return logit, proba


def inception_v4_stem(stacker, name='inception_v4_stem'):
    with tf.variable_scope(name):
        def mix_0(x, name='mix_0'):
            with tf.variable_scope(name):
                a = max_pooling(x, CONV_FILTER_3322, name='a')
                b = conv_block(x, 96, CONV_FILTER_3322, relu, name='b')
                return concat([a, b], axis=3)

        stacker.conv_block(32, CONV_FILTER_3322, relu)
        stacker.conv_block(32, CONV_FILTER_3311, relu)
        stacker.conv_block(64, CONV_FILTER_3311, relu)
        stacker.add_layer(mix_0)

        def mix_1(x, name='mix_1'):
            with tf.variable_scope(name):
                a = conv_block(x, 64, CONV_FILTER_1111, relu, name='a0')
                a = conv_block(a, 96, CONV_FILTER_3311, relu, name='a1')

                b = conv_block(x, 64, CONV_FILTER_1111, relu, name='b0')
                b = conv_block(b, 64, CONV_FILTER_1711, relu, name='b1')
                b = conv_block(b, 64, CONV_FILTER_7111, relu, name='b2')
                b = conv_block(b, 96, CONV_FILTER_3311, relu, name='b3')

                return concat([a, b], axis=3)

        stacker.add_layer(mix_1)

        def mix_2(x, name='mix_2'):
            with tf.variable_scope(name):
                a = conv_block(x, 196, CONV_FILTER_3322, relu, name='a0')
                b = max_pooling(x, CONV_FILTER_3322, name='b0')

                return concat([a, b], axis=3)

        stacker.add_layer(mix_2)
        return stacker


def inception_v4_A(x, name='inception_v4_A'):
    with tf.variable_scope(name):
        a = conv_block(x, 96, CONV_FILTER_1111, relu, name='a0')

        b = conv_block(x, 64, CONV_FILTER_1111, relu, name='b0')
        b = conv_block(b, 96, CONV_FILTER_3311, relu, name='b1')

        c = conv_block(x, 64, CONV_FILTER_1111, relu, name='c0')
        c = conv_block(c, 96, CONV_FILTER_3311, relu, name='c1')
        c = conv_block(c, 96, CONV_FILTER_3311, relu, name='c2')

        d = avg_pooling(x, CONV_FILTER_3311, name='d0')
        d = conv_block(d, 96, CONV_FILTER_1111, relu, name='d1')

        return concat([a, b, c, d], axis=3)


def inception_v4_B(x, name='inception_v4_B'):
    with tf.variable_scope(name):
        a = conv_block(x, 384, CONV_FILTER_1111, relu, name='a0')

        b = conv_block(x, 192, CONV_FILTER_1111, relu, name='b0')
        b = conv_block(b, 224, CONV_FILTER_7111, relu, name='b1')
        b = conv_block(b, 256, CONV_FILTER_1711, relu, name='b2')

        c = conv_block(x, 192, CONV_FILTER_1111, relu, name='c0')
        c = conv_block(c, 224, CONV_FILTER_7111, relu, name='c1')
        c = conv_block(c, 224, CONV_FILTER_1711, relu, name='c2')
        c = conv_block(c, 224, CONV_FILTER_7111, relu, name='c3')
        c = conv_block(c, 256, CONV_FILTER_1711, relu, name='c4')

        d = avg_pooling(x, CONV_FILTER_3311, name='d0')
        d = conv_block(d, 128, CONV_FILTER_1111, relu, name='d1')

        return concat([a, b, c, d], axis=3)


def inception_v4_C(x, name='inception_v4_B'):
    with tf.variable_scope(name):
        a = conv_block(x, 256, CONV_FILTER_1111, relu, name='a0')

        b = conv_block(x, 384, CONV_FILTER_1111, relu, name='b0')
        b0 = conv_block(b, 256, CONV_FILTER_3111, relu, name='b1-0')
        b1 = conv_block(b, 256, CONV_FILTER_1311, relu, name='b1-1')

        c = conv_block(x, 384, CONV_FILTER_1111, relu, name='c0')
        c = conv_block(c, 448, CONV_FILTER_3111, relu, name='c1')
        c = conv_block(c, 512, CONV_FILTER_1311, relu, name='c2')
        c0 = conv_block(c, 256, CONV_FILTER_3111, relu, name='c3-0')
        c1 = conv_block(c, 256, CONV_FILTER_1311, relu, name='c3-1')

        d = avg_pooling(x, CONV_FILTER_3311, name='d0')
        d = conv_block(d, 256, CONV_FILTER_1111, relu, name='d1')

        return concat([a, b0, b1, c0, c1, d], axis=3)


def inception_v4_reduction_A(x, name='inception_v4_reduction_A'):
    with tf.variable_scope(name):
        a = conv_block(x, 384, CONV_FILTER_3322, relu, name='a0')

        b = conv_block(x, 192, CONV_FILTER_1111, relu, name='b0')
        b = conv_block(b, 224, CONV_FILTER_3311, relu, name='b1')
        b = conv_block(b, 256, CONV_FILTER_3322, relu, name='b2')

        c = max_pooling(x, CONV_FILTER_3322, name='c0')

        return concat([a, b, c], axis=3)


def inception_v4_reduction_B(x, name='inception_v4_reduction_B'):
    with tf.variable_scope(name):
        a = conv_block(x, 192, CONV_FILTER_1111, relu, name='a0')
        a = conv_block(a, 192, CONV_FILTER_3322, relu, name='a1')

        b = conv_block(x, 256, CONV_FILTER_1111, relu, name='b0')
        b = conv_block(b, 256, CONV_FILTER_7111, relu, name='b1')
        b = conv_block(b, 320, CONV_FILTER_1711, relu, name='b2')
        b = conv_block(b, 320, CONV_FILTER_3322, relu, name='b3')

        c = max_pooling(x, CONV_FILTER_3322, name='c0')

        return concat([a, b, c], axis=3)


def inception_resnet_v1_stem():
    pass


def inception_resnet_V2_stem():
    pass


def inception_v4_aux(x, n_classes, name='inception_v4_aux'):
    with tf.variable_scope(name):
        stack = Stacker(x)
        stack.avg_pooling(CONV_FILTER_5533)
        stack.conv_block(128, CONV_FILTER_1111, relu)
        filter_ = list(stack.last_layer.shape[1:3]) + [1, 1]
        stack.conv_block(768, filter_, relu)
        stack.flatten()
        logit = stack.linear(n_classes)
        proba = stack.softmax()
        return logit, proba


class InceptionStructure(Base_net_structure):

    def __init__(self, x, n_classes, model_type=3, reuse=False, name=None, verbose=0):
        super().__init__(reuse, name, verbose)
        self.x = x
        self.n_classes = n_classes
        self.model_type = model_type
        self.build_funcs = {
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

        stacker.add_layer(inception_v1_dimension_reduction, channels_3a)
        stacker.add_layer(inception_v1_dimension_reduction, channels_3b)
        stacker.max_pooling(CONV_FILTER_3322)

        stacker.add_layer(inception_v1_dimension_reduction, channels_4a)
        self.aux0_logit, self.aux0_proba = self.aux(stacker.last_layer, 64, 'aux0')
        stacker.add_layer(inception_v1_dimension_reduction, channels_4b)
        stacker.add_layer(inception_v1_dimension_reduction, channels_4c)
        stacker.add_layer(inception_v1_dimension_reduction, channels_4d)
        self.aux1_logit, self.aux1_proba = self.aux(stacker.last_layer, 64, 'aux1')
        stacker.add_layer(inception_v1_dimension_reduction, channels_4e)
        stacker.max_pooling(CONV_FILTER_3322)

        stacker.add_layer(inception_v1_dimension_reduction, channels_5a)
        stacker.add_layer(inception_v1_dimension_reduction, channels_5b)
        stacker.max_pooling(CONV_FILTER_7777)

        # dropout
        stacker.flatten()
        stacker.linear_block(1000, relu)
        stacker.linear(self.n_classes)
        logit = stacker.last_layer
        stacker.softmax()
        proba = stacker.last_layer
        return logit, proba

    def stem_v2(self, stacker, name='stem_v2'):
        Stacker()
        # stacker = Stacker(x)
        with tf.variable_scope(name):
            stacker.conv_block(32, CONV_FILTER_3322, relu)
            stacker.conv_block(32, CONV_FILTER_3311, relu)

            stacker.conv_block(64, CONV_FILTER_3311, relu)
            stacker.max_pooling(CONV_FILTER_3322)

            stacker.conv_block(80, CONV_FILTER_1111, relu)
            stacker.conv_block(192, CONV_FILTER_3311, relu)
            stacker.max_pooling(CONV_FILTER_3322)

            return stacker

    def v2(self, stacker):
        stacker.resize_image((299, 299))

        stacker = self.stem_v2(stacker)

        a_channels0 = {
            'a0': 64,
            'b0': 48,
            'b1': 64,
            'c0': 64,
            'c1': 96,
            'c2': 96,
            'd1': 32
        }
        a_channels1 = {
            'a0': 64,
            'b0': 48,
            'b1': 64,
            'c0': 64,
            'c1': 96,
            'c2': 96,
            'd1': 64
        }
        a_channels2 = {
            'a0': 64,
            'b0': 48,
            'b1': 64,
            'c0': 64,
            'c1': 96,
            'c2': 96,
            'd1': 64
        }
        stacker.add_layer(inception_v2_module_a, a_channels0)
        stacker.add_layer(inception_v2_module_a, a_channels1)
        stacker.add_layer(inception_v2_module_a, a_channels2)

        a_b_multi_pool_channels = {
            'a0': 384,

            'b0': 64,
            'b1': 96,
            'b2': 96,

            'c0': 64,
        }
        stacker.add_layer(inception_v2_module_a_b_multi_pool, a_b_multi_pool_channels)

        b_channels0 = {
            'a0': 192,

            'b0': 128,
            'b1': 128,
            'b2': 192,

            'c0': 128,
            'c1': 128,
            'c2': 128,
            'c3': 128,
            'c4': 192,

            'd1': 192,
        }
        b_channels1 = {
            'a0': 192,

            'b0': 160,
            'b1': 160,
            'b2': 192,

            'c0': 160,
            'c1': 160,
            'c2': 160,
            'c3': 160,
            'c4': 192,

            'd1': 192,
        }
        b_channels2 = {
            'a0': 192,

            'b0': 160,
            'b1': 160,
            'b2': 192,

            'c0': 160,
            'c1': 160,
            'c2': 160,
            'c3': 160,
            'c4': 192,

            'd1': 192,
        }
        b_channels3 = {
            'a0': 192,

            'b0': 192,
            'b1': 192,
            'b2': 192,

            'c0': 192,
            'c1': 192,
            'c2': 192,
            'c3': 192,
            'c4': 192,

            'd1': 192,
        }
        stacker.add_layer(inception_v2_module_b, b_channels0)
        stacker.add_layer(inception_v2_module_b, b_channels1)
        stacker.add_layer(inception_v2_module_b, b_channels2)
        stacker.add_layer(inception_v2_module_b, b_channels3)
        self.aux_logit, self.aux_proba = inception_v2_aux(stacker.last_layer, self.n_classes)

        b_c_multipool_channels = {
            'a0': 192,
            'a1': 320,

            'b0': 192,
            'b1': 192,
            'b2': 192,
            'b3': 192,
        }
        stacker.add_layer(inception_v2_module_b_c_multipool, b_c_multipool_channels)

        c_channels0 = {
            'a0': 320,

            'b0': 384,
            'b1-0': 384,
            'b1-1': 384,

            'c0': 448,
            'c1': 384,
            'c2-0': 384,
            'c2-1': 384,

            'd1': 192,
        }
        c_channels1 = {
            'a0': 320,

            'b0': 384,
            'b1-0': 384,
            'b1-1': 384,

            'c0': 448,
            'c1': 384,
            'c2-0': 384,
            'c2-1': 384,

            'd1': 192,
        }
        stacker.add_layer(inception_v2_module_c, c_channels0)
        stacker.add_layer(inception_v2_module_c, c_channels1)

        # inception refactorize * 3
        # inception refactorize asymetric 6 * 5
        # inception 7 * 2

        stacker.max_pooling((8, 8, 8, 8))
        # dropout
        stacker.flatten()
        stacker.linear_block(1000, relu)
        stacker.linear(self.n_classes)
        logit = stacker.last_layer
        stacker.softmax()
        proba = stacker.last_layer
        return logit, proba

    def build(self):
        with tf.variable_scope(self.name):
            self.stacker = Stacker(self.x)

            self.logit, self.proba = self.build_funcs[self.model_type](self.stacker)

    def v3(self):
        pass

    def v4(self, stacker):
        stacker.resize_image((299, 299))
        stacker = inception_v4_stem(stacker)

        for i in range(4):
            stacker.add_layer(inception_v4_A)

        stacker.add_layer(inception_v4_reduction_A)
        for i in range(7):
            stacker.add_layer(inception_v4_B)
        self.aux_logit, self.aux_proba = inception_v4_aux(stacker.last_layer, self.n_classes)

        stacker.add_layer(inception_v4_reduction_B)

        for i in range(3):
            stacker.add_layer(inception_v4_C)

        # stem
        # A * 4
        # reduction A
        # b * 7
        # reduction B
        # C * 3
        # avg pool

        stacker.max_pooling((8, 8, 8, 8))
        stacker.flatten()
        stacker.linear_block(1000, relu)
        stacker.linear(self.n_classes)
        logit = stacker.last_layer
        stacker.softmax()
        proba = stacker.last_layer
        return logit, proba
