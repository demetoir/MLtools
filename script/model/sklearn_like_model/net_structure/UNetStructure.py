import tensorflow as tf
from script.model.sklearn_like_model.net_structure.Base_net_structure import Base_net_structure
from script.util.Stacker import Stacker
from script.util.tensor_ops import CONV_FILTER_3311, relu, CONV_FILTER_2211, pixel_wise_softmax, CONV_FILTER_2222


class UNetStructure(Base_net_structure):

    def __init__(self, Xs, level=4, n_classes=2, reuse=False, name='Unet'):
        super().__init__(reuse, name)

        self.X = Xs
        self.level = level
        self.n_classes = n_classes
        self.stacker = Stacker(Xs)

    def build(self):
        self.logit, self.proba = self.Unet_recursion_build(
            self.stacker, self.level, self.n_classes, self.reuse, self.name)

    @staticmethod
    def Unet_recursion_build(stacker, level=4, n_classes=2, reuse=False, name='Unet'):
        def _Unet_recursion(stacker, n_channel, level):
            if level == 0:
                stacker.conv_block(n_channel, CONV_FILTER_3311, relu)
                stacker.conv_block(n_channel, CONV_FILTER_3311, relu)
            else:
                # encode
                stacker.conv_block(n_channel, CONV_FILTER_3311, relu)
                stacker.conv_block(n_channel, CONV_FILTER_3311, relu)
                concat = stacker.last_layer
                stacker.max_pooling(CONV_FILTER_2222)

                stacker = _Unet_recursion(stacker, n_channel * 2, level - 1)

                # decode
                stacker.upscale_2x_block(n_channel, CONV_FILTER_2211, relu)

                stacker.concat(concat, axis=3)
                stacker.conv_block(n_channel, CONV_FILTER_3311, relu)
                stacker.conv_block(n_channel, CONV_FILTER_3311, relu)

            return stacker

        with tf.variable_scope(name, reuse=reuse):
            stacker = _Unet_recursion(stacker, n_channel=64, level=level)
            stacker.conv_block(n_classes, CONV_FILTER_3311, relu)
            logit = stacker.last_layer
            proba = pixel_wise_softmax(logit)

        return logit, proba
