"""operation util for tensorflow"""

import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod

"""convolution filter option
(kernel height, kernel width, stride height, stride width)
"""
CONV_FILTER_1111 = (1, 1, 1, 1)
CONV_FILTER_1122 = (1, 1, 2, 2)
CONV_FILTER_2211 = (2, 2, 1, 1)
CONV_FILTER_2222 = (2, 2, 2, 2)
CONV_FILTER_3311 = (3, 3, 1, 1)
CONV_FILTER_3111 = (3, 1, 1, 1)
CONV_FILTER_1311 = (1, 3, 1, 1)
CONV_FILTER_3322 = (3, 3, 2, 2)
CONV_FILTER_4411 = (4, 4, 1, 1)
CONV_FILTER_4422 = (4, 4, 2, 2)
CONV_FILTER_5511 = (5, 5, 1, 1)
CONV_FILTER_5522 = (5, 5, 2, 2)
CONV_FILTER_5533 = (5, 5, 3, 3)
CONV_FILTER_6611 = (6, 6, 1, 1)
CONV_FILTER_6622 = (6, 6, 2, 2)
CONV_FILTER_7711 = (7, 7, 1, 1)
CONV_FILTER_7722 = (7, 7, 2, 2)
CONV_FILTER_7777 = (7, 7, 7, 7)
CONV_FILTER_7111 = (7, 1, 1, 1)
CONV_FILTER_1711 = (1, 7, 1, 1)
CONV_FILTER_9911 = (9, 9, 1, 1)
CONV_FILTER_9922 = (9, 9, 2, 2)


# normalization
def bn(x, is_training=True, name='bn'):
    """batch normalization layer"""
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=name)


class SingletonNameScope:
    @property
    def name_counts(self):
        cls = self.__class__

        if not hasattr(cls, 'name_count'):
            setattr(cls, 'name_count', {})

        return getattr(cls, 'name_count')

    @property
    def name_dict(self):
        cls = self.__class__
        if not hasattr(cls, 'name_dict'):
            setattr(cls, 'name_dict', {})
        return getattr(cls, 'name_dict')

    def new_name(self, name):
        name_counts = self.name_counts
        if name not in name_counts:
            name_counts[name] = 0

        name_counts[name] += 1
        return name + str(name_counts[name])


class tensor_op:
    def __init__(self, name=None, reuse=None):
        if name is None:
            name = self.__class__.__name__
        self._name = name
        self._reuse = reuse
        self.is_build = False
        self.singleton_name_scope = SingletonNameScope()
        self.name_scope = None

    @property
    def __name__(self):
        return self.name

    @property
    def name(self):
        return self._name

    @property
    def reuse(self):
        return self._reuse

    def call(self, inputs, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        self.name_scope = self.singleton_name_scope.new_name(join_scope(get_scope(), self.name))

        with tf.variable_scope(self.name_scope, self.reuse):
            self.node = self.call(*args, **kwargs)

        self.is_build = True
        return self.node


class SingletonBN:
    @property
    def set_train_ops(self):
        cls = self.__class__

        if not hasattr(cls, '__set_train_ops'):
            setattr(cls, '__set_train_ops', {})

        return getattr(cls, '__set_train_ops')

    @property
    def set_non_train_ops(self):
        cls = self.__class__

        if not hasattr(cls, '__set_non_train_ops'):
            setattr(cls, '__set_non_train_ops', {})

        return getattr(cls, '__set_non_train_ops')

    @property
    def is_train_var_collection(self):
        cls = self.__class__

        if not hasattr(cls, '__is_train_var_collection'):
            setattr(cls, '__is_train_var_collection', {})

        return getattr(cls, '__is_train_var_collection')

    def collect_set_train_ops(self, scope):
        set_train_ops = self.set_train_ops
        ops = [
            v
            for k, v in set_train_ops.items()
            if scope in k
        ]
        return ops

    def collect_set_non_train_ops(self, scope):
        set_non_train_ops = self.set_non_train_ops
        ops = [
            v
            for k, v in set_non_train_ops.items()
            if scope in k
        ]
        return ops

    def add_set_train_op(self, op):
        set_train_ops = self.set_train_ops
        set_train_ops[op.name] = op

    def add_set_non_train_op(self, op):
        set_non_train_ops = self.set_non_train_ops
        set_non_train_ops[op.name] = op

    def get_is_train_variable(self):
        head = split_scope(get_scope())[0]

        var_collection = self.is_train_var_collection
        if head not in var_collection:
            is_train_var = tf.Variable(True, dtype=tf.bool, name=f'{head}_BN_is_train')
            set_train_op = tf.assign(is_train_var, True, name=f'{head}_BN_set_train_op')
            set_non_train_op = tf.assign(is_train_var, False, name=f'{head}_BN_set_non_train_op')

            self.add_set_non_train_op(set_non_train_op)
            self.add_set_train_op(set_train_op)
            var_collection[head] = is_train_var

        return var_collection[head]


class BN(tensor_op):
    def __init__(self, momentum=0.99, trainable=True, name=None, reuse=None):
        super().__init__(name, reuse)
        self.singleton_bn = SingletonBN()
        self.momentum = momentum
        self.trainable = trainable

    def call(self, inputs, *args, **kwargs):
        self.is_train_var = self.singleton_bn.get_is_train_variable()

        bn = tf.layers.batch_normalization(
            inputs,
            momentum=self.momentum,
            training=self.is_train_var,
        )
        return bn


class Sigmoid(tensor_op):
    def call(self, inputs, *args, **kwargs):
        return tf.sigmoid(inputs)


class LRelu(tensor_op):
    def __init__(self, alpha=0.1, name=None, reuse=None):
        super().__init__(name, reuse)
        self.alpha = alpha

    def call(self, inputs, *args, **kwargs):
        return tf.maximum(inputs, self.alpha * inputs)


class Elu(tensor_op):
    def call(self, inputs, *args, **kwargs):
        return tf.nn.elu(features=inputs)


class Tanh(tensor_op):
    def call(self, inputs, *args, **kwargs):
        return tf.tanh(inputs)


class Relu(tensor_op):
    def call(self, inputs, *args, **kwargs):
        return tf.nn.relu(inputs)


class Softmax(tensor_op):
    def call(self, inputs, *args, **kwargs):
        return tf.nn.softmax(inputs)


class Linear(tensor_op):

    def __init__(self, output_size, init_bias=0.0, name=None, reuse=None):
        super().__init__(name, reuse)
        self.output_size = output_size
        self.init_bias = init_bias

    def call(self, inputs, *args, **kwargs):
        shape = inputs.get_shape().as_list()

        weight = tf.get_variable(
            "weight",
            [shape[1], self.output_size],
            tf.float32,
            initializer=tf.contrib.layers.xavier_initializer()
        )
        bias = tf.get_variable(
            "bias",
            [self.output_size],
            initializer=tf.constant_initializer(self.init_bias)
        )
        return tf.matmul(inputs, weight) + bias


class Conv2d(tensor_op):
    def __init__(self, filters, kernel, stride=(1, 1), padding='SAME', name=None, reuse=None):
        super().__init__(name, reuse)
        self.filters = filters
        self.kernel = kernel
        self.padding = padding
        self.stride = stride

    def call(self, inputs, *args, **kwargs):
        return tf.layers.conv2d(inputs, self.filters, self.kernel, self.stride, self.padding)


class ConvTranspose2d(tensor_op):
    def __init__(self, filters, kernel, stride=(1, 1), padding='valid', name=None, reuse=None):
        super().__init__(name, reuse)
        self.filters = filters
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def call(self, inputs, *args, **kwargs):
        return tf.layers.conv2d_transpose(inputs, self.filters, self.stride, self.padding)


class MaxPooling2d(tensor_op):
    def __init__(self, pool_size, strides, padding='valid', name=None, reuse=None):
        super().__init__(name, reuse)
        self.padding = padding
        self.strides = strides
        self.pool_size = pool_size

    def call(self, inputs, *args, **kwargs):
        return tf.layers.max_pooling2d(inputs, self.pool_size, self.strides, self.padding)


class AveragePooling2d(tensor_op):
    def __init__(self, pool_size, strides, padding='valid', name=None, reuse=None):
        super().__init__(name, reuse)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def call(self, inputs, *args, **kwargs):
        return tf.layers.average_pooling2d(inputs, self.pool_size, self.strides, self.padding)


class Flatten(tensor_op):
    def call(self, inputs, *args, **kwargs):
        tf.layers.flatten(inputs)


class SingletonDropout:
    @property
    def set_train_ops(self):
        cls = self.__class__

        if not hasattr(cls, '__set_train_ops'):
            setattr(cls, '__set_train_ops', {})

        return getattr(cls, '__set_train_ops')

    @property
    def set_non_train_ops(self):
        cls = self.__class__

        if not hasattr(cls, '__set_non_train_ops'):
            setattr(cls, '__set_non_train_ops', {})

        return getattr(cls, '__set_non_train_ops')

    def collect_set_train_ops(self, scope):
        set_train_ops = self.set_train_ops
        ops = [
            v
            for k, v in set_train_ops.items()
            if scope in k
        ]
        return ops

    def collect_set_non_train_ops(self, scope):
        set_non_train_ops = self.set_non_train_ops
        ops = [
            v
            for k, v in set_non_train_ops.items()
            if scope in k
        ]
        return ops

    def add_set_train_op(self, op):
        set_train_ops = self.set_train_ops
        set_train_ops[op.name] = op

    def add_set_non_train_op(self, op):
        set_non_train_ops = self.set_non_train_ops
        set_non_train_ops[op.name] = op

    @property
    def is_train_var_collection(self):
        cls = self.__class__

        if not hasattr(cls, '__is_train_var_collection'):
            setattr(cls, '__is_train_var_collection', {})

        return getattr(cls, '__is_train_var_collection')

    def get_is_train_variable(self):
        head = split_scope(get_scope())[0]
        var_collection = self.is_train_var_collection
        if head not in var_collection:
            is_train_var = tf.Variable(True, dtype=tf.bool, name=f'{head}_Dropout_is_train')
            set_train_op = tf.assign(is_train_var, True, name=f'{head}_Dropout_set_train_op')
            set_non_train_op = tf.assign(is_train_var, False, name=f'{head}_Dropout_set_non_train_op')

            self.add_set_non_train_op(set_non_train_op)
            self.add_set_train_op(set_train_op)
            var_collection[head] = is_train_var

        return var_collection[head]


class Dropout(tensor_op):
    def __init__(self, dropout_rate, name=None, reuse=None):
        super().__init__(name, reuse)
        self.dropout_rate = dropout_rate
        self.singletonDropout = SingletonDropout()

    def call(self, inputs, *args, **kwargs):
        self.is_train_var = self.singletonDropout.get_is_train_variable()
        dropout = tf.layers.dropout(inputs, self.dropout_rate, training=self.is_train_var)
        return dropout


# activation function
def sigmoid(x, name='sigmoid'):
    """sigmoid activation function layer"""
    return tf.sigmoid(x, name=name)


def tanh(x, name='tanh'):
    """tanh activation function layer"""
    return tf.tanh(x, name=name)


def lrelu(x, leak=0.2, name="lrelu"):
    """leak relu activate function layer"""
    return tf.maximum(x, leak * x, name=name)


def relu(input_, name='relu'):
    """relu activate function layer"""
    return tf.nn.relu(features=input_, name=name)


def elu(input_, name="elu"):
    """elu activate function layer"""
    return tf.nn.elu(features=input_, name=name)


activation_names = ['none', 'sigmoid', 'tanh', 'relu', 'lrelu', 'elu']
name_to_activation = {
    'sigmoid': sigmoid,
    'tanh': tanh,
    'lrelu': lrelu,
    'relu': relu,
    'elu': elu,
    'none': tf.identity,
    'relu6': tf.nn.relu6
}


def linear(input_, output_size, name="linear", stddev=0.02, bias_start=0.0, with_w=False):
    """pre-activated linear layer

    typical one layer of neural net, return just before activate
    input * weight + bias

    :type input_: Union[tensorflow.Variable, tensorflow.PlaceHolder]
    :type output_size: int
    :type name: str
    :type stddev: float
    :type bias_start: float
    :type with_w: bool
    :param input_: input variable or placeholder of tensorflow
    :param output_size: output layer size
    :param name: tensor scope name
    :param stddev: stddev for initialize weight
    :param bias_start: initial value of baise
    :param with_w: return with weight and bias tensor variable

    :return: before activate neural net
    :rtype tensorflow.Variable

    """
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        weight = tf.get_variable("weight", [shape[1], output_size], tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, weight) + bias, weight, bias
        else:
            return tf.matmul(input_, weight) + bias


def conv2d_transpose(input_, output_shape, filter_, name="conv2d_transpose", stddev=0.02,
                     padding='SAME'):
    """transposed 2d convolution layer

    :type input_: Union[tensorflow.Variable, tensorflow.PlaceHolder]
    :type output_shape: Union[list, tuple]
    :type filter_: tuple[int, int, int, int]
    :type name: str
    :type stddev: float
    :param input_: input variable or placeholder of tensorflow
    :param output_shape: output shape of after transposed convolution
    :param filter_: convolution filter(kernel and stride)
    :param name: tensor scope name
    :param stddev: stddev for initialize weight

    :return: result of 2d transposed convolution
    :rtype tensorflow.Variable
    """
    k_h, k_w, d_h, d_w = filter_
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv_transpose = tf.nn.conv2d_transpose(
            input_, weight, output_shape=output_shape, strides=[1, d_h, d_w, 1],
            padding=padding)

        bias = tf.get_variable('bias', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        conv_transpose = tf.nn.bias_add(conv_transpose, bias)

        return conv_transpose


def conv2d(input_, output_channel, filter_, stddev=0.02, name="conv2d"):
    """2d convolution layer

    :type input_: Union[tensorflow.Variable, tensorflow.PlaceHolder]
    :type output_channel: int
    :type filter_: tuple[int, int, int, int]
    :type name: str
    :type stddev: float
    :param input_: input variable or placeholder of tensorflow
    :param output_channel: output shape of after convolution
    :param filter_: convolution filter(kernel and stride)
    :param name: tensor scope name
    :param stddev: stddev for initialize weight

    :return: result of 2d convolution
    :rtype tensorflow.Variable
    """
    k_h, k_w, d_h, d_w = filter_
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [k_h, k_w, input_.get_shape()[-1], output_channel],
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        # initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, weight, strides=[1, d_h, d_w, 1], padding='SAME')

        bias = tf.get_variable('bias', [output_channel], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, bias)

        return conv


def atrous_conv2d(input_, output_channel, filter_, rate, name='atrous_conv2d'):
    k_h, k_w, d_h, d_w = filter_
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [k_h, k_w, input_.get_shape()[-1], output_channel],
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.atrous_conv2d(input_, weight, rate, padding='SAME')

        bias = tf.get_variable('bias', [output_channel], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, bias)

        return conv


def atrous_conv2d_block(input_, output_channel, filter_, rate, activation, name='atrous_conv2d_block'):
    with tf.variable_scope(name):
        net = atrous_conv2d(input_, output_channel, filter_, rate)
        net = bn(net)
        net = activation(net)
    return net


def atrous_conv2d_transpose(input_, output_shape, filter_, rate, name='atrous_conv2d_transpose'):
    k_h, k_w, d_h, d_w = filter_
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv_transpose = tf.nn.atrous_conv2d_transpose(
            input_, weight, output_shape=output_shape, rate=rate, padding='SAME')

        bias = tf.get_variable('bias', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        conv_transpose = tf.nn.bias_add(conv_transpose, bias)

        return conv_transpose


def atrous_conv2d_transpose_block(input_, output_shape, filter_, rate, activation,
                                  name='atrous_conv2d_transpose_block'):
    with tf.variable_scope(name):
        net = atrous_conv2d_transpose(input_, output_shape, filter_, rate)
        net = bn(net)
        net = activation(net)
    return net


def conv2d_one_by_one(input_, output_channel, name='conv2d_one_by_one'):
    """bottle neck convolution layer

    1*1 kernel 1*1 stride convolution

    :type input_: Union[tensorflow.Variable, tensorflow.PlaceHolder]
    :type output_channel: int
    :type name: str
    :param input_: input variable or placeholder of tensorflow
    :param output_channel: output shape of after convolution
    :param name: tensor scope name

    :return: bottle neck convolution
    :rtype tensorflow.Variable
    """
    out = conv2d(input_, output_channel, CONV_FILTER_1111, name=name)
    return out


def upscale_2x_atrous(input_, output_channel, filter_, rate, name='upscale_2x_atrous'):
    shape = input_.get_shape()
    n, h, w, c = list(shape)
    if n.value is None:
        n = -1
    else:
        n = int(n)
    output_shape = [n, int(h) * 2, int(w) * 2, int(output_channel)]

    with tf.variable_scope(name):
        return atrous_conv2d_transpose(input_, output_shape, filter_, rate)


def upscale_2x_atrous_block(input_, output_channel, filter_, rate, activation, name='upscale_2x_atrous_block'):
    with tf.variable_scope(name):
        net = upscale_2x_atrous(input_, output_channel, filter_, rate)
        net = bn(net)
        net = activation(net)
    return net


def upscale_2x(input_, output_channel, filter_, padding='SAME', name='upscale_2x'):
    """transposed convolution to double scale up layer

    doubled width and height of input

    :type input_: Union[tensorflow.Variable, tensorflow.PlaceHolder]
    :type output_channel: int
    :type filter_: tuple[int, int, int, int]
    :type name: str
    :param input_: input variable or placeholder of tensorflow
    :param output_channel: output shape of after transposed convolution
    :param filter_: convolution filter(kernel and stride)
    :param name: tensor scope name

    :return: result of 2*2 upscale
    :rtype tensorflow.Variable
    """
    shape = input_.get_shape()
    n, h, w, c = list(shape)
    if n.value is None:
        n = -1

    output_shape = [n, h * 2, w * 2, output_channel]
    output_shape = list(map(int, output_shape))
    return conv2d_transpose(input_, output_shape, filter_, name=name, padding=padding)


def upscale_2x_block(input_, output_channel, filter_, activate, padding='SAME', name='upscale_2x_block'):
    """2*2 upscale tensor block(transposed convolution, batch normalization, activation)

    :type input_: Union[tensorflow.Variable, tensorflow.PlaceHolder]
    :type output_channel: int
    :type filter_: tuple[int, int, int, int]
    :type name: str
    :type activate: func
    :param input_: input variable or placeholder of tensorflow
    :param output_channel: output shape of after transposed convolution
    :param filter_: convolution filter(kernel and stride)
    :param name: tensor scope name
    :param activate: activate function

    :return: result of tensor block
    :rtype tensorflow.Variable
    """
    with tf.variable_scope(name):
        input_ = upscale_2x(input_, output_channel, filter_, padding=padding)
        input_ = bn(input_)
        input_ = activate(input_)
    return input_


def conv_block(input_, output_channel, filter_, activate, name='conv_block'):
    """convolution tensor block(convolution, batch normalization, activation)

    :type input_: Union[tensorflow.Variable, tensorflow.PlaceHolder]
    :type output_channel: int
    :type filter_: tuple[int, int, int, int]
    :type name: str
    :type activate: func
    :param input_: input variable or placeholder of tensorflow
    :param output_channel: output shape of after convolution
    :param filter_: convolution filter(kernel and stride)
    :param name: tensor scope name
    :param activate: activate function

    :return: result of tensor block
    :rtype tensorflow.Variable
    """
    with tf.variable_scope(name):
        input_ = conv2d(input_, output_channel, filter_, name='conv')
        input_ = bn(input_)
        input_ = activate(input_)
    return input_


def linear_block(input_, output_size, activate, name="linear_block"):
    with tf.variable_scope(name):
        input_ = linear(input_, output_size)
        input_ = bn(input_)
        input_ = activate(input_)
    return input_


def avg_pooling(input_, filter_, name='avg_pooling'):
    """average pooling layer

    :type input_: Union[tensorflow.Variable, tensorflow.PlaceHolder]
    :type filter_: tuple[int, int, int, int]
    :type name: str
    :param input_: input variable or placeholder of tensorflow
    :param filter_: pooling filter(kernel and stride)
    :param name: tensor scope name

    :return: result of average pooling
    :rtype tensorflow.Variable
    """
    kH, kW, sH, sW = filter_
    return tf.nn.avg_pool(input_, ksize=[1, kH, kW, 1], strides=[1, sH, sW, 1], padding='SAME', name=name)


def max_pooling(input_, filter_, padding='SAME', name='max_pooling'):
    """max pooling layer

    :type input_: Union[tensorflow.Variable, tensorflow.PlaceHolder]
    :type filter_: tuple[int, int, int, int]
    :type name: str
    :param input_: input variable or placeholder of tensorflow
    :param filter_: pooling filter(kernel and stride)
    :param name: tensor scope name

    :return: result of max pooling
    :rtype tensorflow.Variable
    """
    kH, kW, sH, sW = filter_
    return tf.nn.max_pool(input_, ksize=[1, kH, kW, 1], strides=[1, sH, sW, 1], padding=padding, name=name)


def onehot_to_index(onehot):
    with tf.variable_scope('onehot_to_index'):
        index = tf.cast(tf.argmax(onehot, 1), tf.float32)
    return index


def index_to_onehot(index, size):
    # TODO implement
    onehot = tf.one_hot(index, size)
    return onehot


def softmax(input_, name='softmax'):
    """softmax layer"""
    return tf.nn.softmax(input_, name=name)


def dropout(input_, rate, name="dropout"):
    """dropout"""
    return tf.nn.dropout(input_, rate, name=name)


def L1_norm(var_list, lambda_=1.0, name="L1_norm"):
    return tf.multiply(lambda_,
                       tf.reduce_sum([tf.reduce_sum(tf.abs(var)) for var in var_list]),
                       name=name)


def L2_norm(var_list, lambda_=1.0, name="L2_norm"):
    return tf.multiply(lambda_,
                       tf.sqrt(tf.reduce_sum([tf.reduce_sum(tf.square(tf.abs(var))) for var in var_list])),
                       name=name)


def wall_decay(decay_rate, global_step, wall_step, name='decay'):
    return tf.pow(decay_rate, global_step // wall_step, name=name)


def average_top_k_loss(loss, k, name='average_top_k_loss'):
    values, indices = tf.nn.top_k(loss, k=k, name=name)
    return values


def reshape(input_, shape, name='reshape'):
    if shape[0] is None:
        shape[0] = -1
    return tf.reshape(input_, shape, name=name)


def concat(values, axis, name="concat"):
    return tf.concat(values, axis, name=name)


def flatten(input_, name='flatten'):
    return tf.layers.flatten(input_, name=name)


def join_scope(*args, spliter='/'):
    return spliter.join(map(str, args))


def split_scope(scope, spliter='/'):
    return str(scope).split(spliter)


def get_scope():
    return tf.get_variable_scope().name


def collect_vars(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)


def placeholder(dtype, shape, name):
    if len(shape) == 0:
        shape = []
    elif shape[0] == -1:
        shape[0] = None

    return tf.placeholder(dtype=dtype, shape=shape, name=name)


def identity(input_, name):
    return tf.identity(input_, name)


def tf_minmax_scaling(x, epsilon=1e-7, name='minmax_scaling'):
    with tf.variable_scope(name, ):
        min_ = tf.reduce_min(x)
        max_ = tf.reduce_max(x)
        return (x - min_) / (max_ - min_ + epsilon)


def tf_z_score_normalize(x: tf.Tensor, name='z_score_normalize'):
    with tf.variable_scope(name):
        if len(x.shape) is not 1:
            raise TypeError('x rank must be 1')
        mean, stddev = tf.nn.moments(x, 0)
        return (x - mean) / stddev


def _residual_block(x, weight, activation, batch_norm):
    x_in = x

    x = weight(x)
    if batch_norm:
        x = bn(x)
    x = activation(x)

    x = weight(x)

    x += x_in
    if batch_norm:
        x = bn(x)
    x = activation(x)

    return x


def _residual_block_with_bottle_neck(x, weight, activation, bottle_neck, batch_norm):
    x_in = x

    x = bottle_neck(x)
    if batch_norm:
        x = bn(x)
    x = activation(x)

    x = weight(x)
    if batch_norm:
        x = bn(x)
    x = activation(x)

    x = bottle_neck(x)

    x += x_in
    if batch_norm:
        x = bn(x)
    x = activation(x)

    return x


def _residual_block_full_pre_activation(x, weight, activation):
    x_in = x

    x = bn(x)
    x = activation(x)
    x = weight(x)

    x = bn(x)
    x = activation(x)
    x = weight(x)

    x += x_in

    return x


def residual_block(x, weight, activation, bottle_neck=None, batch_norm=True, full_pre_activation=False,
                   name='residual_block'):
    with tf.variable_scope(name):
        if full_pre_activation:
            return _residual_block_full_pre_activation(x, weight, activation)

        if bottle_neck:
            return _residual_block_with_bottle_neck(x, weight, activation, bottle_neck, batch_norm)
        else:
            return _residual_block(x, weight, activation, batch_norm)


def resize_image(x, shape, method=ResizeMethod.BILINEAR, align_corners=False, preserve_aspect_ratio=False,
                 name='resize_image'):
    with tf.variable_scope(name):
        x_resize = tf.image.resize_images(x, shape, method, align_corners, preserve_aspect_ratio)
    return x_resize


def residual_add(x, x_add, name='residual_add'):
    with tf.variable_scope(name):
        return x + x_add


def pixel_wise_softmax(output_map, name='pixel_wise_softmax', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
        return exponential_map / normalize


def MAE_loss(x, y, name='MAE'):
    return tf.identity(tf.reduce_mean(tf.abs(x - y)), name=name)


def MSE_loss(x, y, name='MSE'):
    return tf.identity(tf.reduce_mean((x - y) * (x - y)), name=name)


def RMSE_loss(x, y, name='RMSE'):
    return tf.identity(tf.sqrt(tf.reduce_mean(x - y) * (x - y)), name=name)
