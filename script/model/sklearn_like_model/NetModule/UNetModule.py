from script.model.sklearn_like_model.NetModule.BaseNetModule import BaseNetModule
from script.model.sklearn_like_model.NetModule.DynamicDropoutRate import DynamicDropoutRate
from script.util.Stacker import Stacker
from script.util.tensor_ops import *


class UNetModule(BaseNetModule):
    def __init__(self, x, n_classes=2, level=4, depth=1, capacity=64, dropout_rate=0.5, reuse=False, name=None,
                 verbose=0):
        super().__init__(capacity=capacity, reuse=reuse, name=name, verbose=verbose)
        self.x = x
        self.n_classes = n_classes
        self.level = level
        self.depth = depth
        self.n_channel = self.capacity
        self.dropout_rate = dropout_rate

    def build(self):
        def _Unet_recursion(stacker, n_channel, level, dropout_tensor=None):
            if level == 0:
                for i in range(self.depth):
                    stacker.conv_block(n_channel, CONV_FILTER_3311, relu)
                if dropout_tensor:
                    stacker.dropout(dropout_tensor)

            else:
                # encode
                for i in range(self.depth):
                    stacker.conv_block(n_channel, CONV_FILTER_3311, relu)
                if dropout_tensor:
                    stacker.dropout(dropout_tensor)

                concat = stacker.last_layer
                stacker.max_pooling(CONV_FILTER_2222)

                stacker = _Unet_recursion(stacker, n_channel * 2, level - 1)

                # decode
                stacker.upscale_2x_block(n_channel, CONV_FILTER_2211, relu)

                stacker.concat(concat, axis=3)
                for i in range(self.depth):
                    stacker.conv_block(n_channel, CONV_FILTER_3311, relu)
                    if dropout_tensor:
                        stacker.dropout(dropout_tensor)

            return stacker

        with tf.variable_scope(self.name, reuse=self.reuse):
            self.DynamicDropoutRate = DynamicDropoutRate(self.dropout_rate)
            self.dropout_tensor = self.DynamicDropoutRate.tensor

            self.stacker = Stacker(self.x, verbose=self.verbose)

            self.stacker = _Unet_recursion(
                self.stacker,
                n_channel=self.n_channel,
                level=self.level,
                dropout_tensor=self.dropout_tensor
            )

            self.stacker.conv2d(self.n_classes, CONV_FILTER_3311)
            self.logit = self.stacker.last_layer

            self.stacker.pixel_wise_softmax()
            self.proba = self.stacker.last_layer

        return self

    def set_train(self, sess):
        self.DynamicDropoutRate.set_train(sess)

    def set_non_train(self, sess):
        self.DynamicDropoutRate.set_non_train(sess)

    def update_dropout_rate(self, sess, x):
        self.dropout_rate = x
        self.DynamicDropoutRate.update(sess, x)
