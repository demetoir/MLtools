from tqdm import trange
from script.model.sklearn_like_model.GAN.GAN_MixIn import GAN_loss_builder_MixIn
from script.model.sklearn_like_model.BaseModel import BaseModel
from script.model.sklearn_like_model.Mixin import Xs_MixIn, zs_MixIn, Ys_MixIn
from script.util.tensor_ops import *
from script.util.Stacker import Stacker
import numpy as np


class TrainFailError(BaseException):
    pass


class InputShapeError(BaseException):
    pass


def generator(z, Y, net_shape, flatten_size, output_shape, reuse=False, name='generator'):
    with tf.variable_scope(name, reuse=reuse):
        layer = Stacker(concat((z, Y), axis=1))

        for shape in net_shape:
            layer.linear_block(shape, lrelu)

        layer.linear(flatten_size)
        layer.sigmoid()
        layer.reshape(output_shape)

    return layer.last_layer


def discriminator(X, Y, net_shapes, reuse=False, name='discriminator'):
    with tf.variable_scope(name, reuse=reuse):
        layer = Stacker(concat([flatten(X), Y], axis=1))

        for shape in net_shapes:
            layer.linear_block(shape, lrelu)

        layer.linear(1)
        layer.sigmoid()

    return layer.last_layer


class basicC_GANPropertyMixIN(Xs_MixIn, Ys_MixIn, zs_MixIn):
    def __init__(self):
        Xs_MixIn.__init__(self)
        Ys_MixIn.__init__(self)
        zs_MixIn.__init__(self)

    @property
    def _train_ops(self):
        with_clipping = getattr(self, 'with_clipping', False)

        train_ops = [
            getattr(self, 'train_G'),
            getattr(self, 'train_D'),
            getattr(self, 'op_inc_global_step')
        ]

        if with_clipping:
            train_ops += [getattr(self, 'clip_D_op')]

        return train_ops

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


class C_GAN(BaseModel, basicC_GANPropertyMixIN, GAN_loss_builder_MixIn):
    _params_keys = [
        'n_noise',
        'batch_size',
        'learning_rate',
        'D_net_shape',
        'G_net_shape',
        'loss_type',
        'clipping'
    ]

    def __init__(self, n_noise=256, batch_size=64, learning_rate=0.0002, D_net_shape=(256, 256,),
                 G_net_shape=(256, 256,), loss_type='GAN', with_clipping=False, clipping=0.01, verbose=10):
        BaseModel.__init__(self, verbose)
        basicC_GANPropertyMixIN.__init__(self)
        GAN_loss_builder_MixIn.__init__(self)

        self.n_noise = n_noise
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.D_net_shape = D_net_shape
        self.G_net_shape = G_net_shape
        self.loss_type = loss_type
        self.with_clipping = with_clipping
        self.clipping = clipping

    def _build_input_shapes(self, shapes):
        ret = {}
        ret.update(self._build_Xs_input_shape(shapes))

        shapes['zs'] = [None, self.n_noise]
        ret.update(self._build_zs_input_shape(shapes))

        ret.update(self._build_Ys_input_shape(shapes))

        return ret

    def _build_main_graph(self):
        self.Xs = placeholder(tf.float32, self.Xs_shape, name='Xs')
        self.Ys = placeholder(tf.float32, self.Ys_shape, name='Ys')
        self.zs = placeholder(tf.float32, self.zs_shape, name='zs')

        self.G = generator(self.zs, self.Ys, self.G_net_shape, self.X_flatten_size, self.Xs_shape)
        self.Xs_gen = identity(self.G, 'Xs_gen')
        self.D_real = discriminator(self.Xs, self.Ys, self.D_net_shape)
        self.D_gen = discriminator(self.Xs_gen, self.Ys, self.D_net_shape, reuse=True)

        self.G_vals = collect_vars(join_scope(get_scope(), 'generator'))
        self.D_vals = collect_vars(join_scope(get_scope(), 'discriminator'))

    def _build_loss_function(self):
        self.D_real_loss, self.D_gen_loss, self.D_loss, self.G_loss = \
            self._build_GAN_loss(self.D_real, self.D_gen, self.loss_type)

        self.D_loss_mean = tf.reduce_mean(self.D_loss)
        self.G_loss_mean = tf.reduce_mean(self.G_loss)

    def _build_train_ops(self):
        self.train_D = tf.train.AdamOptimizer(self.learning_rate) \
            .minimize(self.D_loss, var_list=self.D_vals)

        self.train_G = tf.train.AdamOptimizer(self.learning_rate) \
            .minimize(self.G_loss, var_list=self.G_vals)

        if self.with_clipping:
            self.clip_D_op = [var.assign(tf.clip_by_value(var, -self.clipping, self.clipping)) for var in self.D_vals]
        else:
            self.clip_D_op = None

    def train(self, Xs, Ys, epoch=1, save_interval=None, batch_size=None):
        self._prepare_train(Xs=Xs, Ys=Ys)
        dataset = self.to_dummyDataset(Xs=Xs, Ys=Ys)

        if batch_size is None:
            batch_size = self.batch_size

        iter_num = 0
        iter_per_epoch = dataset.size // batch_size
        self.log.info("train epoch {}, iter/epoch {}".format(epoch, iter_per_epoch))
        for e in trange(epoch):
            dataset.shuffle()

            total_G = 0
            total_D = 0
            for i in range(iter_per_epoch):
                iter_num += 1

                Xs, Ys = dataset.next_batch(batch_size)
                zs = self.get_z_rand_normal([batch_size, self.n_noise])
                self.run_ops(self._train_ops, {self._Xs: Xs, self._zs: zs, self._Ys: Ys})
                # self.sess.run(self._train_ops, feed_dict={self._Xs: Xs, self._zs: zs, self._Ys: Ys})

                loss_D, loss_G = self.sess.run([self.D_loss_mean, self.G_loss_mean],
                                               feed_dict={self._Xs: Xs, self._zs: zs, self._Ys: Ys})

                # print("e:{}  D={}, G={}".format(e, loss_D, loss_G))
                if np.isnan(loss_D) or np.isnan(loss_G):
                    self.log.error('loss is nan D loss={}, G loss={}'.format(loss_D, loss_G))
                    raise TrainFailError('loss is nan D loss={}, G loss={}'.format(loss_D, loss_G))

                self.log.info("D={} G={}".format(loss_D, loss_G))
                total_D += loss_D / iter_per_epoch
                total_G += loss_G / iter_per_epoch

            self.log.info("e:{}  D={}, G={}".format(e, total_D, total_G))
            print("e:{}  D={}, G={}".format(e, total_D, total_G))

            if save_interval is not None and e % save_interval == 0:
                self.save()

    def generate(self, size, Ys):
        zs = self.get_z_rand_normal([size, self.n_noise])
        return self.get_tf_values(self._gen_ops, {self.zs: zs, self._Ys: Ys})

    def metric(self, Xs, Ys):
        zs = self.get_z_rand_normal([Xs.shape[0], self.n_noise])
        D_loss, G_loss = self.get_tf_values(self._metric_ops, {self._Xs: Xs, self._zs: zs, self._Ys: Ys})
        return {'D_loss': D_loss, 'G_loss': G_loss}
