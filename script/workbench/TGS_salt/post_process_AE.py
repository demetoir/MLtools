from script.model.sklearn_like_model.BaseModel import BaseModel
from script.model.sklearn_like_model.Mixin import SupervisedMetricCallback
from script.model.sklearn_like_model.NetModule.BaseNetModule import BaseNetModule
from script.model.sklearn_like_model.PlaceHolderModule import PlaceHolderModule
from script.model.sklearn_like_model.TFDynamicLearningRate import TFDynamicLearningRate
from script.util.Stacker import Stacker
from script.util.tensor_ops import *


class conv_encoder_module(BaseNetModule):
    def __init__(self, x, capacity=64, reuse=False, name=None, verbose=0):
        super().__init__(capacity, reuse, name, verbose)

        self.x = x

    def build(self):
        with tf.variable_scope(self.name, reuse=self.reuse):
            self.stacker = Stacker(self.x)

            self.stacker.conv_block(self.capacity, CONV_FILTER_3311, relu)
            self.stacker.conv_block(self.capacity, CONV_FILTER_3311, relu)
            self.stacker.max_pooling(CONV_FILTER_2222)

            self.stacker.conv_block(self.capacity * 2, CONV_FILTER_3311, relu)
            self.stacker.conv_block(self.capacity * 2, CONV_FILTER_3311, relu)
            self.stacker.max_pooling(CONV_FILTER_2222)

            self.stacker.conv_block(self.capacity * 4, CONV_FILTER_3311, relu)
            self.stacker.conv_block(self.capacity * 4, CONV_FILTER_3311, relu)
            self.stacker.max_pooling(CONV_FILTER_2222)

            self.stacker.conv_block(self.capacity * 8, CONV_FILTER_3311, relu)
            self.stacker.conv_block(self.capacity * 8, CONV_FILTER_3311, relu)
            self.stacker.conv_block(self.capacity * 8, CONV_FILTER_3311, relu)

            self.encode = self.stacker.last_layer

        return self


class conv_decoder_module(BaseNetModule):
    def __init__(self, x, output_shape, capacity=64, reuse=False, name=None, verbose=0):
        super().__init__(capacity, reuse, name, verbose)
        self.x = x
        self.output_shape = output_shape
        self.output_channel = output_shape[-1]

    def build(self):
        with tf.variable_scope(self.name, reuse=self.reuse):
            self.stacker = Stacker(self.x)

            self.stacker.conv_block(self.capacity * 8, CONV_FILTER_3311, relu)
            self.stacker.conv_block(self.capacity * 8, CONV_FILTER_3311, relu)
            self.stacker.upscale_2x_block(self.capacity * 8, CONV_FILTER_3322, relu)

            self.stacker.conv_block(self.capacity * 4, CONV_FILTER_3311, relu)
            self.stacker.conv_block(self.capacity * 4, CONV_FILTER_3311, relu)
            self.stacker.upscale_2x_block(self.capacity * 4, CONV_FILTER_3322, relu)

            self.stacker.conv_block(self.capacity * 2, CONV_FILTER_3311, relu)
            self.stacker.conv_block(self.capacity * 2, CONV_FILTER_3311, relu)
            self.stacker.upscale_2x_block(self.capacity * 2, CONV_FILTER_3322, relu)

            self.stacker.conv_block(self.capacity, CONV_FILTER_3311, relu)
            self.stacker.conv_block(self.capacity, CONV_FILTER_3311, relu)
            self.stacker.conv_block(self.capacity, CONV_FILTER_3311, relu)
            self.stacker.conv_block(self.capacity, CONV_FILTER_3311, relu)
            self.stacker.conv2d(self.output_channel, CONV_FILTER_3311)
            self.stacker.relu()

            self.decode = self.stacker.last_layer

        return self


class post_process_AE(BaseModel):
    def __init__(
            self,
            learning_rate=0.01,
            beta1=0.9,
            capacity=64,
            batch_size=100,
            dropout_rate=0.5,
            verbose=10,
            **kwargs
    ):
        BaseModel.__init__(self, verbose, **kwargs)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.dropout_rate = dropout_rate
        self.capacity = capacity

    def _build_input_shapes(self, shapes):
        self.x_ph_module = PlaceHolderModule(shapes['x'], tf.float32, name='x')
        self.y_ph_module = PlaceHolderModule(shapes['y'], tf.float32, name='y')

        ret = {}
        ret.update(self.x_ph_module.shape_dict)
        ret.update(self.y_ph_module.shape_dict)

        return ret

    def _build_main_graph(self):
        self.Xs_ph = self.x_ph_module.build().placeholder
        self.Ys_ph = self.y_ph_module.build().placeholder

        self.encoder_module = conv_encoder_module(self.Xs_ph)
        self.encoder_module.build()
        self.encode = self.encoder_module.encode
        self.decoder_module = conv_decoder_module(self.encode, self.Xs_ph.shape)
        self.decoder_module.build()
        self.decode = self.decoder_module.decode
        self._recon = self.decode

        self.vars = self.encoder_module.vars
        self.vars += self.decoder_module.vars

    def _build_loss_function(self):
        self.loss = tf.squared_difference(self.Ys_ph, self._recon, name='loss')
        self.loss_mean = tf.reduce_mean(self.loss, name='loss_mean')

    def _build_train_ops(self):
        self.drl = TFDynamicLearningRate(self.learning_rate)
        self.drl.build()

        self.train_op = tf.train.AdamOptimizer(
            self.drl.learning_rate, self.beta1
        ).minimize(
            loss=self.loss_mean, var_list=self.vars
        )

    def _train_iter(self, dataset, batch_size):
        x, y = dataset.next_batch(self.batch_size)
        self.sess.run(self.train_op, {self.Xs_ph: x, self.Ys_ph: y})

    def metric(self, x, y):
        if not getattr(self, '_metric_callback', None):
            self._metric_callback = SupervisedMetricCallback(
                self, self.loss_mean, self.Xs_ph, self.Ys_ph
            )

        return self._metric_callback(x, y)

    def update_learning_rate(self, lr):
        self.learning_rate = lr

        if self.sess is not None:
            self.drl.update(self.sess, self.learning_rate)
