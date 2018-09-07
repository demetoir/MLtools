from script.model.sklearn_like_model.net_structure.InceptionSructure.BaseInceptionStructure import \
    BaseInceptionStructure
from script.util.Stacker import Stacker
from script.util.tensor_ops import *


class InceptionV4Structure(BaseInceptionStructure):
    @staticmethod
    def stem(stacker, name='stem'):
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

    @staticmethod
    def inception_A(x, name='inception_A'):
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

    @staticmethod
    def inception_B(x, name='inception_B'):
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

    @staticmethod
    def inception_C(x, name='inception_C'):
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

    @staticmethod
    def reduction_A(x, name='reduction_A'):
        with tf.variable_scope(name):
            a = conv_block(x, 384, CONV_FILTER_3322, relu, name='a0')

            b = conv_block(x, 192, CONV_FILTER_1111, relu, name='b0')
            b = conv_block(b, 224, CONV_FILTER_3311, relu, name='b1')
            b = conv_block(b, 256, CONV_FILTER_3322, relu, name='b2')

            c = max_pooling(x, CONV_FILTER_3322, name='c0')

            return concat([a, b, c], axis=3)

    @staticmethod
    def reduction_B(x, name='reduction_B'):
        with tf.variable_scope(name):
            a = conv_block(x, 192, CONV_FILTER_1111, relu, name='a0')
            a = conv_block(a, 192, CONV_FILTER_3322, relu, name='a1')

            b = conv_block(x, 256, CONV_FILTER_1111, relu, name='b0')
            b = conv_block(b, 256, CONV_FILTER_7111, relu, name='b1')
            b = conv_block(b, 320, CONV_FILTER_1711, relu, name='b2')
            b = conv_block(b, 320, CONV_FILTER_3322, relu, name='b3')

            c = max_pooling(x, CONV_FILTER_3322, name='c0')

            return concat([a, b, c], axis=3)

    @staticmethod
    def aux(x, n_classes, name='aux'):
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

    def structure(self, stacker):
        stacker.resize_image((299, 299))
        stacker = self.stem(stacker)

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

            self.logit, self.proba = self.structure(self.stacker)
