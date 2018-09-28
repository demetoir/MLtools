import numpy as np
from tqdm import tqdm, trange
from script.data_handler.Base.BaseDataset import BaseDataset
from script.model.sklearn_like_model.BaseModel import BaseModel
from script.model.sklearn_like_model.Mixin import UnsupervisedMetricCallback
from script.model.sklearn_like_model.NetModule.BaseNetModule import BaseNetModule
from script.model.sklearn_like_model.NetModule.FusionNetStructure import FusionNetModule
from script.model.sklearn_like_model.PlaceHolderModule import PlaceHolderModule
from script.model.sklearn_like_model.TFDynamicLearningRate import TFDynamicLearningRate
from script.util.Stacker import Stacker
from script.util.tensor_ops import *


class pre_train_Unet(BaseModel):
    def __init__(
            self,
            verbose=10,
            learning_rate=0.01,
            beta1=0.9,
            batch_size=100,
            stage=4,
            n_classes=2,
            capacity=64,
            depth=1,
            dropout_rate=0.5,
            **kwargs
    ):
        BaseModel.__init__(self, verbose, **kwargs)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.dropout_rate = dropout_rate
        self.capacity = capacity

        self.stage = stage
        self.n_classes = n_classes
        self.depth = depth

    def _build_input_shapes(self, shapes):
        self.x_ph_module = PlaceHolderModule(shapes['x'], tf.float32, name='x')

        ret = {}
        ret.update(self.x_ph_module.shape_dict)

        return ret

    def _build_main_graph(self):
        self.Xs = self.x_ph_module.build().placeholder

        self.net_module = FusionNetModule(
            self.Xs, capacity=self.capacity, depth=self.depth, level=self.stage,
            n_classes=self.n_classes, dropout_rate=self.dropout_rate
        ).build()
        self.decode = self.net_module.decode

        self.recon_module = reconModule(
            self.decode, self.capacity
        )
        self.recon_module.build()
        self._recon = self.recon_module.recon
        self._recon = self.decode

        self.vars = self.net_module.vars
        self.vars += self.recon_module.vars

    def _build_loss_function(self):
        self.loss = tf.squared_difference(self.Xs, self._recon, name='loss')
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
        # self.net_module.set_train(self.sess)

        x = dataset.next_batch(self.batch_size)
        _ = self.sess.run(self.train_op, {self.Xs: x})

        # self.net_module.set_predict(self.sess)

    def train_AE(
            self, x, epoch=1, batch_size=None, dataset_callback=None,
            epoch_pbar=True, iter_pbar=True, epoch_callbacks=None,
    ):
        if not self.is_built:
            raise RuntimeError(f'{self} not built')

        batch_size = getattr(self, 'batch_size') if batch_size is None else batch_size
        dataset = dataset_callback if dataset_callback else BaseDataset(x=x)

        metric = None
        epoch_pbar = tqdm([i for i in range(1, epoch + 1)]) if epoch_pbar else None
        for _ in range(1, epoch + 1):
            dataset.shuffle()

            iter_pbar = trange if iter_pbar else range
            for _ in iter_pbar(int(dataset.size / batch_size)):
                self._train_iter(dataset, batch_size)

            self.sess.run(self.op_inc_global_epoch)
            global_epoch = self.sess.run(self.global_epoch)
            if epoch_pbar: epoch_pbar.update(1)

            metric = getattr(self, 'metric', None)(x)
            if metric in (np.nan, np.inf, -np.inf):
                tqdm.write(f'train fail, e = {global_epoch}, metric = {metric}')
                break

            results = []
            if epoch_callbacks:
                for callback in epoch_callbacks:
                    result = callback(self, dataset, metric, global_epoch)
                    results += [result]

            break_epoch = False
            for result in results:
                if result and getattr(result, 'break_epoch', False):
                    break_epoch = True
            if break_epoch: break

        if epoch_pbar: epoch_pbar.close()
        if dataset_callback: del dataset

        return metric

    def metric(self, x):
        if not getattr(self, '_metric_callback', None):
            self._metric_callback = UnsupervisedMetricCallback(
                self, self.loss_mean, self.Xs,
            )

        return self._metric_callback(x)

    def update_learning_rate(self, lr):
        self.learning_rate = lr

        if self.sess is not None:
            self.drl.update(self.sess, self.learning_rate)


class reconModule(BaseNetModule):
    def __init__(self, x, capacity=None, reuse=False, name=None, verbose=0):
        super().__init__(capacity, reuse, name, verbose)
        self.x = x

    def build(self):
        with tf.variable_scope(self.name):
            stacker = Stacker(self.x)
            stacker.conv2d(1, CONV_FILTER_3311)
            self.recon = stacker.sigmoid()

        return self
