import tensorflow as tf

from script.model.sklearn_like_model.net_structure.Base_net_structure import Base_net_structure
from script.util.Stacker import Stacker
from script.util.tensor_ops import CONV_FILTER_3311, relu, CONV_FILTER_2211, CONV_FILTER_2222


class UNetStructure(Base_net_structure):

    def __init__(self, x, n_classes=2, level=4, depth=1, capacity=64, reuse=False, name=None, verbose=0):
        super().__init__(capacity=capacity, reuse=reuse, name=name, verbose=verbose)
        self.x = x
        self.n_classes = n_classes
        self.level = level
        self.depth = depth
        self.n_channel = self.capacity

    def build(self):
        def _Unet_recursion(stacker, n_channel, level):
            if level == 0:
                for i in range(self.depth):
                    stacker.conv_block(n_channel, CONV_FILTER_3311, relu)
            else:
                # encode
                for i in range(self.depth):
                    stacker.conv_block(n_channel, CONV_FILTER_3311, relu)
                concat = stacker.last_layer
                stacker.max_pooling(CONV_FILTER_2222)

                stacker = _Unet_recursion(stacker, n_channel * 2, level - 1)

                # decode
                stacker.upscale_2x_block(n_channel, CONV_FILTER_2211, relu)

                stacker.concat(concat, axis=3)
                for i in range(self.depth):
                    stacker.conv_block(n_channel, CONV_FILTER_3311, relu)

            return stacker

        with tf.variable_scope(self.name, reuse=self.reuse):
            self.stacker = Stacker(self.x, verbose=self.verbose)

            self.stacker = _Unet_recursion(self.stacker, n_channel=self.n_channel, level=self.level)

            self.stacker.conv2d(self.n_classes, CONV_FILTER_3311)
            self.logit = self.stacker.last_layer

            self.stacker.pixel_wise_softmax()
            self.proba = self.stacker.last_layer
