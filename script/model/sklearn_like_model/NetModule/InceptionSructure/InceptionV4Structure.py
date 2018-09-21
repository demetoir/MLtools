from script.model.sklearn_like_model.NetModule.InceptionSructure.BaseInceptionNetModule import \
    BaseInceptionNetModule
from script.util.Stacker import Stacker
from script.util.tensor_ops import *


class InceptionV4NetModule(BaseInceptionNetModule):
    def stem(self, stacker, name='stem'):
        with tf.variable_scope(name):
            def mix_0(x, name='mix_0'):
                with tf.variable_scope(name):
                    a = max_pooling(x, CONV_FILTER_3322, name='a')
                    b = conv_block(x, self.n_channel * 6, CONV_FILTER_3322, relu, name='b')
                    return concat([a, b], axis=3)

            stacker.conv_block(self.n_channel * 2, CONV_FILTER_3322, relu)
            stacker.conv_block(self.n_channel * 2, CONV_FILTER_3311, relu)
            stacker.conv_block(self.n_channel * 4, CONV_FILTER_3311, relu)
            stacker.add_layer(mix_0)

            def mix_1(x, name='mix_1'):
                with tf.variable_scope(name):
                    a = conv_block(x, self.n_channel * 4, CONV_FILTER_1111, relu, name='a0')
                    a = conv_block(a, self.n_channel * 6, CONV_FILTER_3311, relu, name='a1')

                    b = conv_block(x, self.n_channel * 4, CONV_FILTER_1111, relu, name='b0')
                    b = conv_block(b, self.n_channel * 4, CONV_FILTER_1711, relu, name='b1')
                    b = conv_block(b, self.n_channel * 4, CONV_FILTER_7111, relu, name='b2')
                    b = conv_block(b, self.n_channel * 6, CONV_FILTER_3311, relu, name='b3')

                    return concat([a, b], axis=3)

            stacker.add_layer(mix_1)

            def mix_2(x, name='mix_2'):
                with tf.variable_scope(name):
                    a = conv_block(x, 196, CONV_FILTER_3322, relu, name='a0')
                    b = max_pooling(x, CONV_FILTER_3322, name='b0')

                    return concat([a, b], axis=3)

            stacker.add_layer(mix_2)
            return stacker

    def inception_A(self, x, name='inception_A'):
        with tf.variable_scope(name):
            a = conv_block(x, self.n_channel * 6, CONV_FILTER_1111, relu, name='a0')

            b = conv_block(x, self.n_channel * 4, CONV_FILTER_1111, relu, name='b0')
            b = conv_block(b, self.n_channel * 6, CONV_FILTER_3311, relu, name='b1')

            c = conv_block(x, self.n_channel * 4, CONV_FILTER_1111, relu, name='c0')
            c = conv_block(c, self.n_channel * 6, CONV_FILTER_3311, relu, name='c1')
            c = conv_block(c, self.n_channel * 6, CONV_FILTER_3311, relu, name='c2')

            d = avg_pooling(x, CONV_FILTER_3311, name='d0')
            d = conv_block(d, self.n_channel * 6, CONV_FILTER_1111, relu, name='d1')

            return concat([a, b, c, d], axis=3)

    def inception_B(self, x, name='inception_B'):
        with tf.variable_scope(name):
            a = conv_block(x, self.n_channel * 24, CONV_FILTER_1111, relu, name='a0')

            b = conv_block(x, self.n_channel * 12, CONV_FILTER_1111, relu, name='b0')
            b = conv_block(b, self.n_channel * 14, CONV_FILTER_7111, relu, name='b1')
            b = conv_block(b, self.n_channel * 16, CONV_FILTER_1711, relu, name='b2')

            c = conv_block(x, self.n_channel * 12, CONV_FILTER_1111, relu, name='c0')
            c = conv_block(c, self.n_channel * 14, CONV_FILTER_7111, relu, name='c1')
            c = conv_block(c, self.n_channel * 14, CONV_FILTER_1711, relu, name='c2')
            c = conv_block(c, self.n_channel * 14, CONV_FILTER_7111, relu, name='c3')
            c = conv_block(c, self.n_channel * 16, CONV_FILTER_1711, relu, name='c4')

            d = avg_pooling(x, CONV_FILTER_3311, name='d0')
            d = conv_block(d, self.n_channel * 8, CONV_FILTER_1111, relu, name='d1')

            return concat([a, b, c, d], axis=3)

    def inception_C(self, x, name='inception_C'):
        with tf.variable_scope(name):
            a = conv_block(x, self.n_channel * 16, CONV_FILTER_1111, relu, name='a0')

            b = conv_block(x, self.n_channel * 24, CONV_FILTER_1111, relu, name='b0')
            b0 = conv_block(b, self.n_channel * 16, CONV_FILTER_3111, relu, name='b1-0')
            b1 = conv_block(b, self.n_channel * 16, CONV_FILTER_1311, relu, name='b1-1')

            c = conv_block(x, self.n_channel * 24, CONV_FILTER_1111, relu, name='c0')
            c = conv_block(c, self.n_channel * 28, CONV_FILTER_3111, relu, name='c1')
            c = conv_block(c, self.n_channel * 32, CONV_FILTER_1311, relu, name='c2')
            c0 = conv_block(c, self.n_channel * 16, CONV_FILTER_3111, relu, name='c3-0')
            c1 = conv_block(c, self.n_channel * 16, CONV_FILTER_1311, relu, name='c3-1')

            d = avg_pooling(x, CONV_FILTER_3311, name='d0')
            d = conv_block(d, self.n_channel * 16, CONV_FILTER_1111, relu, name='d1')

            return concat([a, b0, b1, c0, c1, d], axis=3)

    def reduction_A(self, x, name='reduction_A'):
        with tf.variable_scope(name):
            a = conv_block(x, self.n_channel * 24, CONV_FILTER_3322, relu, name='a0')

            b = conv_block(x, self.n_channel * 12, CONV_FILTER_1111, relu, name='b0')
            b = conv_block(b, self.n_channel * 14, CONV_FILTER_3311, relu, name='b1')
            b = conv_block(b, self.n_channel * 16, CONV_FILTER_3322, relu, name='b2')

            c = max_pooling(x, CONV_FILTER_3322, name='c0')

            return concat([a, b, c], axis=3)

    def reduction_B(self, x, name='reduction_B'):
        with tf.variable_scope(name):
            a = conv_block(x, self.n_channel * 12, CONV_FILTER_1111, relu, name='a0')
            a = conv_block(a, self.n_channel * 12, CONV_FILTER_3322, relu, name='a1')

            b = conv_block(x, self.n_channel * 16, CONV_FILTER_1111, relu, name='b0')
            b = conv_block(b, self.n_channel * 16, CONV_FILTER_7111, relu, name='b1')
            b = conv_block(b, self.n_channel * 20, CONV_FILTER_1711, relu, name='b2')
            b = conv_block(b, self.n_channel * 20, CONV_FILTER_3322, relu, name='b3')

            c = max_pooling(x, CONV_FILTER_3322, name='c0')

            return concat([a, b, c], axis=3)

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

    def build(self):
        with tf.variable_scope(self.name):
            self.stacker = Stacker(self.x)

            self.stacker.resize_image((299, 299))
            stacker = self.stem(self.stacker)

            for i in range(4):
                stacker.add_layer(self.inception_A)

            stacker.add_layer(self.reduction_A)
            for i in range(7):
                stacker.add_layer(self.inception_B)
            self.aux_logit, self.aux_proba = self.aux(stacker.last_layer, self.n_classes)

            stacker.add_layer(self.reduction_B)

            for i in range(3):
                stacker.add_layer(self.inception_C)

            stacker.max_pooling((8, 8, 8, 8))

            # dropout
            self.flatten_layer = stacker.flatten()
            stacker.linear_block(self.n_channel * 64, relu)
            self.logit = stacker.linear(self.n_classes)
            self.proba = stacker.softmax()
