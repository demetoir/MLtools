from script.model.sklearn_like_model.NetModule.BaseNetModule import BaseNetModule
from script.util.Stacker import Stacker
from script.util.tensor_ops import *


class BaseResNetNetModule(BaseNetModule):
    def __init__(self, x, n_classes, capacity=None, reuse=False, name=None, verbose=0):
        super().__init__(capacity, reuse, name, verbose)
        self.x = x
        self.n_classes = n_classes
        if self.capacity:
            self.n_channel = self.capacity
        else:
            self.n_channel = 64

    @staticmethod
    def residual_block_type2(x, n_channel, batch_norm=True, down_sample=False, name='residual_block_type2'):
        with tf.variable_scope(name):
            x_in = x

            if down_sample:
                x = conv_block(x, n_channel, CONV_FILTER_3322, relu, name='conv1')
            else:
                x = conv_block(x, n_channel, CONV_FILTER_3311, relu, name='conv1')

            if batch_norm:
                x = bn(x, name='bn1')
            x = relu(x, name='relu1')

            x = conv_block(x, n_channel, CONV_FILTER_3311, relu, name='conv2')
            if not down_sample:
                x += x_in

            if batch_norm:
                x = bn(x, name='bn2')
            x = relu(x, name='relu2')

        return x

    @staticmethod
    def residual_block_type3(x, n_channel, batch_norm=True, down_sample=False, name='residual_block_type3'):
        with tf.variable_scope(name):
            x_in = x

            if down_sample:
                x = conv_block(x, n_channel, CONV_FILTER_1122, relu, name='conv1')
                if batch_norm:
                    x = bn(x, name='bn1')
                x = relu(x, name='relu1')

                x = conv_block(x, n_channel, CONV_FILTER_3311, relu, name='conv2')
                if batch_norm:
                    x = bn(x, name='bn2')
                x = relu(x, name='relu2')

                x = conv_block(x, n_channel * 4, CONV_FILTER_1111, relu, name='conv3')

                if batch_norm:
                    x = bn(x, name='bn3')
                x = relu(x, name='relu3')

                x_in = conv_block(x_in, n_channel * 4, CONV_FILTER_1122, relu, name='conv1_add')
                x += x_in
            else:
                x = conv_block(x, n_channel, CONV_FILTER_1111, relu, name='conv1')
                if batch_norm:
                    x = bn(x, name='bn1')
                x = relu(x, name='relu1')

                x = conv_block(x, n_channel, CONV_FILTER_3311, relu, name='conv2')
                if batch_norm:
                    x = bn(x, name='bn2')
                x = relu(x, name='relu2')

                x = conv_block(x, n_channel * 4, CONV_FILTER_1111, relu, name='conv3')
                if x_in.shape[3] != x.shape[3]:
                    x_in = conv_block(x_in, n_channel * 4, CONV_FILTER_1111, relu, name='conv3_add')
                x += x_in

                if batch_norm:
                    x = bn(x, name='bn3')
                x = relu(x, name='relu3')

        return x

    def head(self, stacker):
        # conv1 to 112
        stacker.conv_block(self.n_channel, CONV_FILTER_7722, relu)

        # conv2_x to 56
        stacker.max_pooling(CONV_FILTER_3322)
        return stacker

    def foot(self, stacker, n_channel, n_classes):
        stacker.avg_pooling(CONV_FILTER_7777)
        self.flatten_layer = stacker.flatten()

        stacker.linear_block(n_channel * 64, relu)
        stacker.linear_block(n_channel * 64, relu)
        self.logit = stacker.linear(n_classes)
        self.proba = stacker.softmax()

    def body(self, stacker):
        raise NotImplementedError

    def build(self):
        with tf.variable_scope(self.name, reuse=self.reuse):
            self.stacker = Stacker(self.x, verbose=self.verbose)
            self.stacker.resize_image((224, 224))
            self.stacker = self.head(self.stacker)
            self.stacker = self.body(self.stacker)
            self.foot(self.stacker, self.n_channel, self.n_classes)
