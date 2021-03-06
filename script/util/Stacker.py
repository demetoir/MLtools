from script.util.MixIn import LoggerMixIn
from script.util.tensor_ops import *


class Stacker(LoggerMixIn):
    """help easily make graph model by stacking layer and naming for tensorflow

    ex)
    stacker = Stacker(input_)
    stacker.add_layer(conv2d, 64, CONV_FILTER_5522)
    stacker.add_layer(bn)
    stacker.add_layer(lrelu)

    stacker.add_layer(conv2d, 128, CONV_FILTER_5522)
    stacker.add_layer(bn)
    stacker.add_layer(lrelu)

    stacker.add_layer(conv2d, 256, CONV_FILTER_5522)
    stacker.add_layer(bn)
    stacker.add_layer(lrelu)
    last_layer = stacker.last_layer
    """

    def __init__(self, start_layer=None, reuse=False, name=None, verbose=0):
        """create SequenceModel

        :param start_layer:the start layer
        :param reuse:reuse option for tensorflow graph
        :param name:prefix name for layer
        """
        LoggerMixIn.__init__(self, verbose)
        self.start_layer = start_layer
        self.reuse = reuse
        self.name = name
        self.last_layer = start_layer
        if self.start_layer is None:
            self.layer_seq = []
        else:
            self.layer_seq = [start_layer]
        self.layer_count = len(self.layer_seq)

        if self.start_layer is None:
            self.build_seq = []
        else:
            self.build_seq = None

    @staticmethod
    def layer_info(layer):
        return f'({layer.op.name}, {layer.shape}, {layer.dtype}'

    @property
    def scope_head(self):
        if self.name:
            return f'{self.name}'
        else:
            return f'layer'

    def add_layer(self, func, *args, **kwargs):
        """add new layer right after last added layer

        :param func: function for tensor layer
        :param args: args for layer
        :param kwargs: kwargs for layer
        :return: added new layer
        """
        if self.start_layer is None:
            self.build_seq += [(func, args, kwargs)]
            return self.build_seq[-1]
        else:
            scope_name = f'{self.scope_head}_{self.layer_count}_{func.__name__}'
            with tf.variable_scope(scope_name, reuse=self.reuse):
                if func == concat:
                    self.last_layer = func(*args, **kwargs)
                else:
                    self.last_layer = func(self.last_layer, *args, **kwargs)

                self.layer_seq += [self.last_layer]

                self.log.info(f'{scope_name}, {self.last_layer.shape}')
                self.layer_count += 1

            return self.last_layer

    def add_stacker(self, stacker):
        scope_name = self.name + '_layer' + str(self.layer_count)
        with tf.variable_scope(scope_name, reuse=self.reuse):
            inner = Stacker(self.last_layer, stacker.reuse, stacker.name, stacker.verbose)
            for func, args, kwarg in stacker.build_seq:
                inner.add_layer(func, *args, **kwarg)

            self.last_layer = inner.last_layer
            self.layer_seq += [inner.last_layer]
            self.layer_count += 1

        self.log.info(self.layer_info(self.last_layer))
        return self.last_layer

    def bn(self):
        """add batch normalization layer"""
        return self.add_layer(bn)

    def sigmoid(self):
        """add sigmoid layer"""
        return self.add_layer(sigmoid)

    def tanh(self):
        return self.add_layer(tanh)

    def lrelu(self, **kwargs):
        """add leaky relu layer"""
        return self.add_layer(lrelu, **kwargs)

    def relu(self):
        """add relu layer"""
        return self.add_layer(relu)

    def elu(self):
        """add elu layer"""
        return self.add_layer(elu)

    def linear(self, output_size):
        """add linear layer"""
        return self.add_layer(linear, output_size)

    def linear_block(self, output_size, activate):
        return self.add_layer(linear_block, output_size, activate)

    def conv2d_transpose(self, output_shape, filter_):
        """add 2d transposed convolution layer"""
        return self.add_layer(conv2d_transpose, output_shape, filter_)

    def conv2d(self, output_channel, filter_):
        """add 2d convolution layer"""
        return self.add_layer(conv2d, output_channel, filter_)

    def atrous_conv2d(self, output_channel, filter_, rate):
        return self.add_layer(atrous_conv2d, output_channel, filter_, rate)

    def atrous_conv2d_block(self, output_channel, filter_, rate, activate):
        return self.add_layer(atrous_conv2d_block, output_channel, filter_, rate, activate)

    def conv2d_one_by_one(self, output_channel):
        """add bottle neck convolution layer"""
        return self.add_layer(conv2d_one_by_one, output_channel)

    def upscale_2x(self, output_channel, filter_):
        """add upscale 2x layer"""
        return self.add_layer(upscale_2x, output_channel, filter_)

    def upscale_2x_block(self, output_channel, filter_, activate):
        """add upscale 2x block layer"""
        return self.add_layer(upscale_2x_block, output_channel, filter_, activate)

    def conv_block(self, output_channel, filter_, activate):
        """add convolution block layer"""
        return self.add_layer(conv_block, output_channel, filter_, activate)

    def avg_pooling(self, filter_):
        """add average pooling layer"""
        return self.add_layer(avg_pooling, filter_)

    def max_pooling(self, filter_, padding='SAME'):
        """add max pooling layer"""
        return self.add_layer(max_pooling, filter_, padding)

    def softmax(self):
        """add softmax layer"""
        return self.add_layer(softmax)

    def dropout(self, rate):
        """add dropout layer"""
        return self.add_layer(dropout, rate)

    def reshape(self, shape):
        return self.add_layer(reshape, shape)

    def concat(self, values, axis):
        return self.add_layer(concat, values, axis)

    def flatten(self):
        return self.add_layer(flatten)

    def activation(self, name):
        return self.add_layer(name_to_activation[name])

    def resize_image(self, shape, **kwargs):
        return self.add_layer(resize_image, shape, **kwargs)

    def residual_add(self, x_add):
        return self.add_layer(residual_add, x_add)

    def pixel_wise_softmax(self):
        return self.add_layer(pixel_wise_softmax)

    def layers_conv2d(self, channel, filter, stride, padding):
        return self.add_layer(tf.layers.conv2d, channel, filter, stride, padding)

    def layers_bn(self):
        return self.add_layer(tf.layers.batch_normalization)

    def layers_dropout(self, rate):
        return self.add_layer(tf.layers.dropout, rate)

    def layers_max_pooling2d(self, pool_size, stride):
        return self.add_layer(tf.layers.max_pooling2d, pool_size, stride)

    def layers_conv2d_transpose(self, channel, filter, stride, padding):
        return self.add_layer(tf.layers.conv2d_transpose, channel, filter, stride, padding)
