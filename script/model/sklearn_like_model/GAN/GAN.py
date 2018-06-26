from script.model.sklearn_like_model.BaseModel import BaseModel
from script.util.Stacker import Stacker
from script.util.tensor_ops import *
from functools import reduce
from tqdm import trange
import numpy as np


class TrainFailError(BaseException):
    pass


class InputShapeError(BaseException):
    pass


class basicGANPropertyMixIN:

    @property
    def _Xs(self):
        return getattr(self, 'Xs', None)

    @property
    def _zs(self):
        return getattr(self, 'zs', None)

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


def flatten_shape(x):
    return reduce(lambda a, b: a * b, x)


class basicGeneratorMixIn:
    @staticmethod
    def generator(z, net_shapes, flatten_size, output_shape, reuse=False, name='generator'):
        with tf.variable_scope(name, reuse=reuse):
            layer = Stacker(z)

            for shape in net_shapes:
                layer.linear(shape)

            layer.linear(flatten_size)
            layer.sigmoid()
            layer.reshape(output_shape)

        return layer.last_layer


class basicDiscriminatorMixIn:
    @staticmethod
    def discriminator(X, net_shapes, reuse=False, name='discriminator'):
        with tf.variable_scope(name, reuse=reuse):
            layer = Stacker(flatten(X))

            for shape in net_shapes:
                layer.linear(shape)

            layer.linear(1)
            layer.sigmoid()

        return layer.last_layer


class GAN_z_noiseMixIn:
    @staticmethod
    def get_z_noise(shape):
        return np.random.uniform(-1.0, 1.0, size=shape)


class GAN(BaseModel, basicGANPropertyMixIN, basicGeneratorMixIn, basicDiscriminatorMixIn, GAN_z_noiseMixIn):
    _input_shape_keys = [
        'X_shape',
        'Xs_shape',
        'X_flatten_size',
        'z_shape',
        'zs_shape',
    ]
    _params_keys = [
        'n_noise',
        'batch_size',
        'learning_rate',
        'D_net_shape',
        'G_net_shape',
    ]

    def __init__(self, verbose=10, **kwargs):
        BaseModel.__init__(self, verbose, **kwargs)
        self.n_noise = 256
        self.batch_size = 64
        self.learning_rate = 0.0002
        self.D_net_shape = (128, 128)
        self.G_net_shape = (128, 128)
        self.D_learning_rate = 0.0001
        self.G_learning_rate = 0.0001

        self.X_shape = None
        self.Xs_shape = None
        self.X_flatten_size = None
        self.z_shape = None
        self.zs_shape = None

    def build_input_shapes(self, shapes):
        X_shape = shapes['Xs']
        Xs_shape = [None] + list(X_shape)

        X_flatten_size = flatten_shape(X_shape)

        z_shape = [self.n_noise]
        zs_shape = [None, self.n_noise]

        ret = {
            'X_shape': X_shape,
            'Xs_shape': Xs_shape,
            'X_flatten_size': X_flatten_size,
            'z_shape': z_shape,
            'zs_shape': zs_shape
        }
        return ret

    def _build_main_graph(self):
        self.Xs = placeholder(tf.float32, self.Xs_shape, "Xs")
        self.zs = placeholder(tf.float32, self.zs_shape, "zs")

        self.G = self.generator(self.zs, self.G_net_shape, self.X_flatten_size, self.Xs_shape)
        self.Xs_gen = identity(self.G, 'Xs_gen')
        self.D_real = self.discriminator(self.Xs, self.D_net_shape)
        self.D_gen = self.discriminator(self.Xs_gen, self.D_net_shape, reuse=True)

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
        shapes = {'Xs': Xs.shape[1:]}
        self._apply_input_shapes(self.build_input_shapes(shapes))
        self.is_built()

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
                zs = self.get_z_noise([batch_size, self.n_noise])
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
        zs = self.get_z_noise([size, self.n_noise])
        return self.get_tf_values(self._gen_ops, {self.zs: zs})

    def metric(self, Xs):
        zs = self.get_z_noise([Xs.shape[0], self.n_noise])
        D_loss, G_loss = self.get_tf_values(self._metric_ops, {self._Xs: Xs, self._zs: zs})
        return {'D_loss': D_loss, 'G_loss': G_loss}
