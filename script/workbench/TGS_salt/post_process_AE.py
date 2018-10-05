from tqdm import tqdm
import numpy as np
from script.model.sklearn_like_model.BaseModel import BaseModel
from script.model.sklearn_like_model.Mixin import SupervisedMetricCallback, slice_np_arr
from script.model.sklearn_like_model.NetModule.BaseNetModule import BaseNetModule
from script.model.sklearn_like_model.NetModule.FusionNetStructure import FusionNetModule
from script.model.sklearn_like_model.NetModule.PlaceHolderModule import PlaceHolderModule
from script.model.sklearn_like_model.NetModule.TFDynamicLearningRate import TFDynamicLearningRate
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


class ReconCallback:
    def __init__(self, model, op, x_ph, **kwargs):
        self.sess = model.sess
        self.batch_size = model.batch_size
        self.op = op
        self.x_ph = x_ph
        self.kwargs = kwargs

    def _metric_batch(self, x):
        return self.sess.run(self.op, feed_dict={self.x_ph: x})

    def __call__(self, x):
        size = len(x)
        if size > self.batch_size:
            tqdm.write('batch metric')
            xs = slice_np_arr(x, self.batch_size)
            metrics = [
                self._metric_batch(x)
                for x in tqdm(xs)
            ]
            return np.concatenate(metrics)
        else:
            return self._metric_batch(x)


class post_process_AE(BaseModel):
    def __init__(
            self,
            learning_rate=0.01,
            beta1=0.9,
            capacity=64,
            batch_size=100,
            dropout_rate=0.5,
            verbose=10,
            resize=None,
            **kwargs
    ):
        BaseModel.__init__(self, verbose, **kwargs)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.dropout_rate = dropout_rate
        self.capacity = capacity
        self.resize = resize

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

        self.UNetModule = FusionNetModule(
            self.Xs_ph,
            capacity=self.capacity,
            dropout_rate=self.dropout_rate,
            depth=2,
            level=3,
            n_classes=1
        ).build()

        # todo may fix, proba does not train able, seems like no gradient flow
        # self._recon = self.UNetModule.proba
        # self._recon = sigmoid(self.UNetModule.logit)
        self._recon = self.UNetModule.logit

        self.vars = self.UNetModule.vars

    def _build_loss_function(self):
        self.loss = tf.squared_difference(self.Ys_ph, self._recon, name='loss')
        self.loss_mean = tf.reduce_mean(self.loss, name='loss_mean')

        # self.loss_mean = tf.reduce_mean(self.loss, name='loss_mean') + average_top_k_loss(
        #     flatten(self.loss), 100) * 0.2

    def _build_train_ops(self):
        self.drl = TFDynamicLearningRate(self.learning_rate)
        self.drl.build()

        self.train_op = tf.train.AdamOptimizer(
            self.drl.learning_rate, self.beta1
        ).minimize(
            loss=self.loss_mean, var_list=self.vars
        )

    def _train_iter(self, dataset, batch_size):
        self.set_train()
        x, y = dataset.next_batch(self.batch_size)
        self.sess.run(self.train_op, {self.Xs_ph: x, self.Ys_ph: y})
        self.set_non_train()

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

    def set_non_train(self):
        self.UNetModule.set_non_train(self.sess)

    def set_train(self):
        self.UNetModule.set_train(self.sess)

    def recon(self, x):
        if not getattr(self, '_recon_callback', None):
            setattr(self, '_recon_callback', ReconCallback(self, self._recon, self.Xs_ph))

        return getattr(self, '_recon_callback')(x)
