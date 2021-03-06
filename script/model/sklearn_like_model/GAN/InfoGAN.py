from script.model.sklearn_like_model.GAN.GAN_MixIn import GAN_loss_builder_MixIn
from script.model.sklearn_like_model.BaseModel import BaseModel
from script.model.sklearn_like_model.Mixin import Xs_MixIn, Ys_MixIn, zs_MixIn, cs_MixIn
from script.util.Stacker import Stacker
from script.util.tensor_ops import *
from tqdm import trange
import numpy as np
import tensorflow as tf


class InfoGANPropertyMixIN(Xs_MixIn, Ys_MixIn, zs_MixIn, cs_MixIn):
    def __init__(self):
        Xs_MixIn.__init__(self)
        Ys_MixIn.__init__(self)
        zs_MixIn.__init__(self)
        cs_MixIn.__init__(self)

    @property
    def _train_ops(self):
        with_clipping = getattr(self, 'with_clipping', False)

        train_ops = [
            getattr(self, 'train_G'),
            getattr(self, 'train_D'),
            getattr(self, 'train_Q'),
            getattr(self, 'op_inc_global_step')
        ]

        if with_clipping:
            train_ops += [getattr(self, 'clip_D_op')]

        return train_ops

    @property
    def _metric_ops(self):
        return [
            self._D_loss,
            self._G_loss,
            self._Q_loss,
            self._Q_discrete_loss,
            self._Q_continuous_loss,
        ]

    @property
    def _D_loss(self):
        return getattr(self, 'D_loss')

    @property
    def _G_loss(self):
        return getattr(self, 'G_loss')

    @property
    def _Q_loss(self):
        return getattr(self, 'Q_loss')

    @property
    def _Q_discrete_loss(self):
        return getattr(self, 'Q_discrete_loss')

    @property
    def _Q_continuous_loss(self):
        return getattr(self, 'Q_continuous_loss')

    @property
    def _generate_ops(self):
        return getattr(self, 'Xs_gen', None)


class InfoGAN(BaseModel, InfoGANPropertyMixIN, GAN_loss_builder_MixIn):

    def __init__(self, batch_size=64, learning_rate=0.0002, n_noise=256, n_c=2, with_D_clip=True, D_clipping=0.1,
                 D_net_shapes=(512, 512), G_net_shapes=(512, 512), Q_net_shape=(512, 512), loss_type='GAN', verbose=10):
        BaseModel.__init__(self, verbose)
        InfoGANPropertyMixIN.__init__(self)
        GAN_loss_builder_MixIn.__init__(self)

        self.n_noise = n_noise
        self.n_c = n_c
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.with_clipping = with_D_clip
        self.clipping = D_clipping
        self.D_net_shapes = D_net_shapes
        self.G_net_shapes = G_net_shapes
        self.Q_net_shapes = Q_net_shape
        self.loss_type = loss_type

        # self.len_discrete_code = self.Y_flatten_size  # categorical distribution (i.e. label)
        # self.len_continuous_code = self.n_c  # gaussian distribution (e.g. rotation, thickness)

    def _build_input_shapes(self, shapes):
        ret = {}
        ret.update(self._build_Xs_input_shape(shapes))

        ret.update(self._build_Ys_input_shape(shapes))

        shapes['zs'] = [None, self.n_noise]
        ret.update(self._build_zs_input_shape(shapes))

        shapes['cs'] = [None, self.n_c]
        ret.update(self._build_cs_input_shape(shapes))

        return ret

    @staticmethod
    def Q_function(Xs_gen, net_shapes, discrete_code_size, continuous_code_size, reuse=False, name='Q_function'):
        with tf.variable_scope(name, reuse=reuse):
            layer = Stacker(Xs_gen)
            for shape in net_shapes:
                layer.linear_block(shape, relu)

            code_logit = layer.linear(discrete_code_size + continuous_code_size)
            code = layer.softmax()

            discrete_code = code[:, :discrete_code_size]
            continuous_code = code[:, discrete_code_size:]
            discrete_code_logit = code_logit[:, :discrete_code_size]
            continuous_code_logit = code_logit[:, discrete_code_size:]

        return discrete_code, discrete_code_logit, continuous_code, continuous_code_logit

    @staticmethod
    def generator(zs, Ys, cs, net_shapes, flatten_size, output_shape, reuse=False, name='generator'):
        with tf.variable_scope(name, reuse=reuse):
            layer = Stacker(concat((zs, Ys, cs), axis=1))

            for shape in net_shapes:
                layer.linear_block(shape, lrelu)

            layer.linear(flatten_size)
            layer.sigmoid()
            layer.reshape(output_shape)

        return layer.last_layer

    @staticmethod
    def discriminator(Xs, net_shapes, reuse=False, name='discriminator'):
        with tf.variable_scope(name, reuse=reuse):
            layer = Stacker(flatten(Xs))

            for shape in net_shapes:
                layer.linear_block(shape, lrelu)

            feature = layer.last_layer

            layer.linear(1)
            layer.sigmoid()

        return layer.last_layer, feature

    def _build_main_graph(self):
        self.Xs = placeholder(tf.float32, self.Xs_shape, name='Xs')
        self.Ys = placeholder(tf.float32, self.Ys_shape, name='Ys')
        self.zs = placeholder(tf.float32, self.zs_shape, name='zs')
        self.cs = placeholder(tf.float32, self.cs_shape, name='cs')

        self.Xs_gen = self.generator(self.zs, self.Ys, self.cs, self.G_net_shapes, self.X_flatten_size, self.Xs_shape)
        self.G = self.Xs_gen

        self.D_real, D_real_feature = self.discriminator(self.Xs, self.D_net_shapes)
        self.D_gen, D_gen_feature = self.discriminator(self.Xs_gen, self.D_net_shapes, reuse=True)

        self.discrete_code, self.discrete_code_logit, self.continuous_code, self.continuous_code_logit = \
            self.Q_function(D_gen_feature, self.Q_net_shapes, self.Y_flatten_size, self.n_c)

        self.G_vals = collect_vars(join_scope(get_scope(), 'generator'))
        self.D_vals = collect_vars(join_scope(get_scope(), 'discriminator'))
        self.Q_vals = collect_vars(join_scope(get_scope(), 'Q_function'))

    def _build_loss_ops(self):

        self.D_real_loss, self.D_gen_loss, self.D_loss, self.G_loss = \
            self._build_GAN_loss(self.D_real, self.D_gen, self.loss_type)

        # discrete code : categorical
        disc_code_est = self.discrete_code_logit
        disc_code_tg = self.Ys
        Q_disc_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=disc_code_est, labels=disc_code_tg)
        self.Q_discrete_loss = identity(Q_disc_loss, 'Q_discrete_loss')

        # continuous code : gaussian
        cont_code_est = self.continuous_code_logit
        cont_code_tg = self.cs
        Q_cont_loss = tf.reduce_sum(tf.square(cont_code_tg - cont_code_est), axis=1)
        self.Q_continuous_loss = identity(Q_cont_loss, 'Q_continuous_loss')

        # get information loss
        self.Q_loss = identity(self.Q_continuous_loss + self.Q_discrete_loss, 'Q_loss')
        # self.loss_Q = Q_disc_loss

        self.D_loss_mean = tf.reduce_mean(self.D_loss)
        self.G_loss_mean = tf.reduce_mean(self.G_loss)
        self.Q_loss_mean = tf.reduce_mean(self.Q_loss)

    def _build_train_ops(self):
        self.train_D = tf.train.AdamOptimizer(learning_rate=self.learning_rate) \
            .minimize(self.D_loss, var_list=self.D_vals)

        self.train_G = tf.train.AdamOptimizer(learning_rate=self.learning_rate) \
            .minimize(self.G_loss, var_list=self.G_vals)

        var_list = self.D_vals + self.G_vals + self.Q_vals
        self.train_Q = tf.train.AdamOptimizer(learning_rate=self.learning_rate) \
            .minimize(self.Q_loss, var_list=var_list)

        if self.with_clipping:
            self.clip_D_op = [var.assign(tf.clip_by_value(var, -self.clipping, self.clipping)) for var in self.D_vals]
        else:
            self.clip_D_op = None

    def train(self, Xs, Ys, epoch=1, save_interval=None, batch_size=None, check_loss=True):
        self._prepare_train(Xs=Xs, Ys=Ys)
        dataset = self.to_dummyDataset(Xs=Xs, Ys=Ys)

        if batch_size is None:
            batch_size = self.batch_size

        iter_per_epoch = dataset.size // batch_size
        self.log.info("train epoch {}, iter/epoch {}".format(epoch, iter_per_epoch))
        for e in trange(epoch):
            dataset.shuffle()

            total_loss = {
                'D': 0.,
                'G': 0.,
                'Q_continuous': 0.,
                'Q_discrete': 0.,
                'Q': 0.
            }

            for _ in range(iter_per_epoch):
                Xs, Ys = dataset.next_batch(batch_size)
                zs = self.get_z_rand_normal([batch_size, self.n_noise])
                cs = self.get_z_rand_normal([batch_size, self.n_c])
                self.run_ops(self._train_ops, {self._Xs: Xs, self._Ys: Ys, self._zs: zs, self._cs: cs})

                loss_pack = self.metric(Xs, Ys, zs, cs)
                if check_loss:
                    self._loss_check(loss_pack)
                total_loss = self.add_loss_pack(total_loss, loss_pack)
                self.log.info(self.format_loss_pack(loss_pack))

            total_loss = self.div_loss_pack(total_loss, iter_per_epoch)
            self.log.info(self.format_loss_pack(total_loss))

            if save_interval is not None and e % save_interval == 0:
                self.save()

    def generate(self, Ys, zs=None, cs=None):
        Ys = np.array(Ys)
        batch_size = Ys.shape[0]

        if zs is None:
            zs = self.get_z_rand_normal([batch_size, self.n_noise])

        if cs is None:
            cs = self.get_c_rand_normal([batch_size, self.n_c])

        return self.get_tf_values(self._generate_ops, {self._zs: zs, self._Ys: Ys, self._cs: cs})

    def metric(self, Xs, Ys, zs=None, cs=None):
        Xs = np.array(Xs)
        batch_size = Xs.shape[0]

        if zs is None:
            zs = self.get_z_rand_normal([batch_size, self.n_noise])

        if cs is None:
            cs = self.get_c_rand_normal([batch_size, self.n_c])

        D_loss, G_loss, Q_loss, Q_discrete_loss, Q_continuous_loss = self.get_tf_values(
            self._metric_ops,
            {self._Xs: Xs, self._zs: zs, self._Ys: Ys, self._cs: cs})
        return {
            'D': D_loss, 'G': G_loss, 'Q': Q_loss,
            'Q_discrete': Q_discrete_loss,
            'Q_continuous': Q_continuous_loss}
