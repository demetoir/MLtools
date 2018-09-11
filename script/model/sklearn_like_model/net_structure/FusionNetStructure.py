import tensorflow as tf
from script.model.sklearn_like_model.net_structure.Base_net_structure import Base_net_structure
from script.util.Stacker import Stacker
from script.util.tensor_ops import conv_block, CONV_FILTER_3311, relu, \
    CONV_FILTER_2211, residual_add, CONV_FILTER_2222, CONV_FILTER_1111


class FusionNetStructure(Base_net_structure):

    def __init__(self, x, level=4, n_classes=2, depth=1, capacity=64, reuse=False, name=None, verbose=0):
        super().__init__(capacity, reuse, name, verbose)

        self.x = x
        self.level = level
        self.n_classes = n_classes
        self.depth = depth
        self.n_channel = capacity

    def build(self):
        self.logit, self.proba = self._recursion_build()

    def _recursion_build(self):
        def _residual_block(x, n_channel, filter_, activation, name='residual_block'):
            with tf.variable_scope(name):
                x_in = x
                x = conv_block(x, n_channel, filter_, activation, name='conv_block1')
                x = conv_block(x, n_channel, filter_, activation, name='conv_block2')
                x = conv_block(x, n_channel, filter_, activation, name='conv_block3')
                x = residual_add(x, x_in)
            return x

        def recursion(stacker, n_channel, level):
            if level == 0:
                stacker.conv_block(n_channel, CONV_FILTER_3311, relu)
                for i in range(self.depth):
                    stacker.add_layer(_residual_block, n_channel, CONV_FILTER_3311, relu)
                stacker.conv_block(n_channel, CONV_FILTER_3311, relu)
            else:
                # encode
                stacker.conv_block(n_channel, CONV_FILTER_3311, relu)
                for i in range(self.depth):
                    stacker.add_layer(_residual_block, n_channel, CONV_FILTER_3311, relu)
                stacker.conv_block(n_channel, CONV_FILTER_3311, relu)
                x_add = stacker.last_layer
                concat = stacker.last_layer

                stacker.max_pooling(CONV_FILTER_2222)

                stacker = recursion(stacker, n_channel * 2, level - 1)

                # decode
                stacker.upscale_2x_block(n_channel, CONV_FILTER_2211, relu)

                # TODO hack to dynamic batch size after conv transpose, concat must need. wtf?
                stacker.concat(concat, axis=3)
                stacker.conv_block(n_channel, CONV_FILTER_1111, relu)
                stacker.residual_add(x_add)

                stacker.conv_block(n_channel, CONV_FILTER_3311, relu)
                for i in range(self.depth):
                    stacker.add_layer(_residual_block, n_channel, CONV_FILTER_3311, relu)
                stacker.conv_block(n_channel, CONV_FILTER_3311, relu)

            return stacker

        self.stacker = Stacker(self.x, verbose=self.verbose)
        with tf.variable_scope(self.name, reuse=self.reuse):
            stacker = recursion(self.stacker, self.n_channel, self.level)

            stacker.conv_block(self.n_classes, CONV_FILTER_3311, relu)
            logit = stacker.last_layer
            stacker.pixel_wise_softmax()
            proba = stacker.last_layer

        return logit, proba
