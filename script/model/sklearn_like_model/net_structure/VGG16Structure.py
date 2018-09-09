import tensorflow as tf

from script.model.sklearn_like_model.net_structure.Base_net_structure import Base_net_structure
from script.util.Stacker import Stacker
from script.util.tensor_ops import CONV_FILTER_3311, relu, CONV_FILTER_2222


class VGG16Structure(Base_net_structure):

    def __init__(self, x, n_classes, capacity=None, reuse=False, name='VGG16Structure', verbose=0):
        super().__init__(capacity, reuse, name, verbose)
        self.x = x
        self.n_classes = n_classes
        if capacity:
            self.n_channel = capacity
        else:
            self.n_channel = 64

    def build(self):
        with tf.variable_scope(self.name):
            self.stacker = Stacker(self.x, verbose=self.verbose)
            #  resize to 224 * 224
            self.stacker.resize_image((224, 224))

            # 224
            self.stacker.conv_block(self.n_channel, CONV_FILTER_3311, relu)
            self.stacker.conv_block(self.n_channel, CONV_FILTER_3311, relu)
            self.stacker.max_pooling(CONV_FILTER_2222)

            # 112
            self.stacker.conv_block(self.n_channel * 2, CONV_FILTER_3311, relu)
            self.stacker.conv_block(self.n_channel * 2, CONV_FILTER_3311, relu)
            self.stacker.max_pooling(CONV_FILTER_2222)

            # 52
            self.stacker.conv_block(self.n_channel * 4, CONV_FILTER_3311, relu)
            self.stacker.conv_block(self.n_channel * 4, CONV_FILTER_3311, relu)
            self.stacker.conv_block(self.n_channel * 4, CONV_FILTER_3311, relu)
            self.stacker.max_pooling(CONV_FILTER_2222)

            # 28
            self.stacker.conv_block(self.n_channel * 8, CONV_FILTER_3311, relu)
            self.stacker.conv_block(self.n_channel * 8, CONV_FILTER_3311, relu)
            self.stacker.conv_block(self.n_channel * 8, CONV_FILTER_3311, relu)
            self.stacker.max_pooling(CONV_FILTER_2222)

            # 14
            self.stacker.conv_block(self.n_channel * 8, CONV_FILTER_3311, relu)
            self.stacker.conv_block(self.n_channel * 8, CONV_FILTER_3311, relu)
            self.stacker.conv_block(self.n_channel * 8, CONV_FILTER_3311, relu)
            self.stacker.max_pooling(CONV_FILTER_2222)

            # 7 512 to fc
            self.stacker.flatten()
            self.stacker.linear_block(self.n_channel * 64, relu)
            self.stacker.linear_block(self.n_channel * 64, relu)
            self.stacker.linear(self.n_classes)
            self.logit = self.stacker.last_layer
            self.stacker.softmax()
            self.proba = self.stacker.last_layer
