from script.model.sklearn_like_model.Mixin import Xs_MixIn, zs_MixIn
from script.model.sklearn_like_model.BaseModel import BaseModel
from script.util.Stacker import Stacker
from script.util.tensor_ops import *
from tqdm import trange
import numpy as np


class TrainFailError(BaseException):
    pass


class InputShapeError(BaseException):
    pass


class basicGANPropertyMixIN(Xs_MixIn, zs_MixIn):
    def __init__(self):
        Xs_MixIn.__init__(self)
        zs_MixIn.__init__(self)

    @property
    def _train_ops(self):
        return [
            getattr(self, 'train_G'),
            getattr(self, 'train_D'),
            getattr(self, 'op_inc_global_step')
        ]

    @property
    def _metric_ops(self):
        return [
            self._loss_D,
            self._loss_G
        ]

    @property
    def _loss_D(self):
        return getattr(self, 'D_loss')

    @property
    def _loss_G(self):
        return getattr(self, 'G_loss')

    @property
    def _gen_ops(self):
        return getattr(self, 'Xs_gen', None)


def basicGenerator(z, net_shapes, flatten_size, output_shape, reuse=False, name='generator'):
    with tf.variable_scope(name, reuse=reuse):
        layer = Stacker(z)

        for shape in net_shapes:
            layer.linear(shape)

        layer.linear(flatten_size)
        layer.sigmoid()
        layer.reshape(output_shape)

    return layer.last_layer


def basicDiscriminator(X, net_shapes, reuse=False, name='discriminator'):
    with tf.variable_scope(name, reuse=reuse):
        layer = Stacker(flatten(X))

        for shape in net_shapes:
            layer.linear(shape)

        layer.linear(1)
        layer.sigmoid()

    return layer.last_layer


class GAN(BaseModel, basicGANPropertyMixIN):
    _params_keys = [
        'n_noise',
        'batch_size',
        'learning_rate',
        'D_net_shape',
        'G_net_shape',
    ]

    def __init__(self, verbose=10, **kwargs):
        BaseModel.__init__(self, verbose, **kwargs)
        basicGANPropertyMixIN.__init__(self)

        self.n_noise = 256
        self.batch_size = 64
        self.learning_rate = 0.0002
        self.D_net_shape = (128, 128)
        self.G_net_shape = (128, 128)
        self.D_learning_rate = 0.0001
        self.G_learning_rate = 0.0001

    def _build_input_shapes(self, shapes):
        ret = {}
        ret.update(self._build_Xs_input_shape(shapes))

        shapes['zs'] = [None, self.n_noise]
        ret.update(self._build_zs_input_shape(shapes))

        return ret

    def _build_main_graph(self):
        self.Xs = placeholder(tf.float32, self.Xs_shape, "Xs")
        self.zs = placeholder(tf.float32, self.zs_shape, "zs")

        self.G = basicGenerator(self.zs, self.G_net_shape, self.X_flatten_size, self.Xs_shape)
        self.Xs_gen = identity(self.G, 'Xs_gen')
        self.D_real = basicDiscriminator(self.Xs, self.D_net_shape)
        self.D_gen = basicDiscriminator(self.Xs_gen, self.D_net_shape, reuse=True)

        self.G_vals = collect_vars(join_scope(get_scope(), 'generator'))
        self.D_vals = collect_vars(join_scope(get_scope(), 'discriminator'))

    def _build_loss_function(self):
        self.D_real_loss = tf.identity(self.D_real, 'loss_D_real')
        self.D_gen_loss = tf.identity(self.D_gen, 'loss_D_gen')

        self.D_loss = tf.identity(-tf.log(self.D_real) - tf.log(1. - self.D_gen), name='loss_D')
        self.G_loss = tf.identity(-tf.log(self.D_gen), name='loss_G')
        self.D_loss_mean = tf.reduce_mean(self.D_loss)
        self.G_loss_mean = tf.reduce_mean(self.G_loss)

    def _build_train_ops(self):
        self.train_D = tf.train.AdamOptimizer(learning_rate=self.D_learning_rate) \
            .minimize(self.D_loss, var_list=self.D_vals)

        self.train_G = tf.train.AdamOptimizer(learning_rate=self.G_learning_rate) \
            .minimize(self.G_loss, var_list=self.G_vals)

    def train(self, Xs, epoch=1, save_interval=None, batch_size=None, shuffle=True):
        self._prepare_train(Xs=Xs)
        dataset = self.to_dummyDataset(Xs=Xs)

        if batch_size is None:
            batch_size = self.batch_size

        iter_num = 0
        iter_per_epoch = dataset.size // batch_size
        self.log.info("train epoch {}, iter/epoch {}".format(epoch, iter_per_epoch))
        for e in trange(epoch):
            if shuffle:
                dataset.shuffle()

            total_G = 0
            total_D = 0
            for i in range(iter_per_epoch):
                iter_num += 1

                Xs = dataset.next_batch(batch_size)
                zs = self.get_z_rand([batch_size, self.n_noise])
                self.sess.run(self._train_ops, feed_dict={self._Xs: Xs, self._zs: zs})

                loss_D, loss_G = self.sess.run([self.D_loss_mean, self.G_loss_mean],
                                               feed_dict={self._Xs: Xs, self._zs: zs})

                if np.isnan(loss_D) or np.isnan(loss_G):
                    self.log.error('loss is nan D loss={}, G loss={}'.format(loss_D, loss_G))
                    raise TrainFailError('loss is nan D loss={}, G loss={}'.format(loss_D, loss_G))

                # self.log.info("D={} G={}".format(loss_D, loss_G))
                total_D += loss_D / iter_per_epoch
                total_G += loss_G / iter_per_epoch

            self.log.info("e:{}  D={}, G={}".format(e, total_D, total_G))

            if save_interval is not None and e % save_interval == 0:
                self.save()

    def generate(self, size):
        zs = self.get_z_rand([size, self.n_noise])
        return self.get_tf_values(self._gen_ops, {self.zs: zs})

    def metric(self, Xs):
        zs = self.get_z_rand([Xs.shape[0], self.n_noise])
        D_loss, G_loss = self.get_tf_values(self._metric_ops, {self._Xs: Xs, self._zs: zs})
        return {'D_loss': D_loss, 'G_loss': G_loss}
