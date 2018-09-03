from script.model.sklearn_like_model.net_structure.Base_net_structure import Base_net_structure
from script.util.Stacker import Stacker
from script.util.tensor_ops import *


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


class ResNetStructure(Base_net_structure):

    def __init__(self, x, n_classes, model_type=18, reuse=False, name=None, verbose=0):
        super().__init__(reuse, name, verbose)
        self.x = x
        self.n_classes = n_classes
        self.model_type = model_type
        self.build_funcs = {
            18: self.type_18,
            34: self.type_34,
            50: self.type_50,
            101: self.type_101,
            152: self.type_152,
        }

    def type_18(self, stacker):
        stacker.add_layer(residual_block_type2, 64)
        stacker.add_layer(residual_block_type2, 64)

        stacker.add_layer(residual_block_type2, 128, down_sample=True)
        stacker.add_layer(residual_block_type2, 128)

        stacker.add_layer(residual_block_type2, 256, down_sample=True)
        stacker.add_layer(residual_block_type2, 256)

        stacker.add_layer(residual_block_type2, 512, down_sample=True)
        stacker.add_layer(residual_block_type2, 512)

        return stacker

    def type_34(self, stacker):
        for i in range(3):
            stacker.add_layer(residual_block_type2, 64)

        stacker.add_layer(residual_block_type2, 128, down_sample=True)
        for i in range(4 - 1):
            stacker.add_layer(residual_block_type2, 128)

        stacker.add_layer(residual_block_type2, 256, down_sample=True)
        for i in range(6 - 1):
            stacker.add_layer(residual_block_type2, 256)

        stacker.add_layer(residual_block_type2, 512, down_sample=True)
        for i in range(3 - 1):
            stacker.add_layer(residual_block_type2, 512)

        return stacker

    def type_50(self, stacker):
        for i in range(3):
            stacker.add_layer(residual_block_type3, 64)

        stacker.add_layer(residual_block_type3, 128, down_sample=True)
        for i in range(4 - 1):
            stacker.add_layer(residual_block_type3, 128)

        stacker.add_layer(residual_block_type3, 256, down_sample=True)
        for i in range(6 - 1):
            stacker.add_layer(residual_block_type3, 256)

        stacker.add_layer(residual_block_type3, 512, down_sample=True)
        for i in range(3):
            stacker.add_layer(residual_block_type3, 512)

        return stacker

    def type_101(self, stacker):
        for i in range(3):
            stacker.add_layer(residual_block_type3, 64)

        stacker.add_layer(residual_block_type3, 128, down_sample=True)
        for i in range(4 - 1):
            stacker.add_layer(residual_block_type3, 128)

        stacker.add_layer(residual_block_type3, 256, down_sample=True)
        for i in range(23 - 1):
            stacker.add_layer(residual_block_type3, 256)

        stacker.add_layer(residual_block_type3, 512, down_sample=True)
        for i in range(3):
            stacker.add_layer(residual_block_type3, 512)

        return stacker

    def type_152(self, stacker):
        for i in range(3):
            stacker.add_layer(residual_block_type3, 64)

        stacker.add_layer(residual_block_type3, 128, down_sample=True)
        for i in range(8 - 1):
            stacker.add_layer(residual_block_type3, 128)

        stacker.add_layer(residual_block_type3, 256, down_sample=True)
        for i in range(36 - 1):
            stacker.add_layer(residual_block_type3, 256)

        stacker.add_layer(residual_block_type3, 512, down_sample=True)
        for i in range(3):
            stacker.add_layer(residual_block_type3, 512)

        return stacker

    def head(self, stacker):
        # conv1 to 112
        stacker.conv_block(64, CONV_FILTER_7722, relu)

        # conv2_x to 56
        stacker.max_pooling(CONV_FILTER_3322)
        return stacker

    def build(self):
        with tf.variable_scope(self.name, reuse=self.reuse):
            self.stacker = Stacker(self.x, verbose=self.verbose)
            self.stacker.resize_image((224, 224))

            self.stacker = self.head(self.stacker)
            build_func = self.build_funcs[self.model_type]
            self.stacker = build_func(self.stacker)

            self.stacker.avg_pooling(CONV_FILTER_7777)
            self.stacker.flatten()
            self.stacker.linear_block(1000, relu)
            self.stacker.linear_block(1000, relu)

            self.stacker.linear(self.n_classes)
            self.logit = self.stacker.last_layer
            self.stacker.softmax()
            self.proba = self.stacker.last_layer
