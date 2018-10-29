from tqdm import trange
from script.model.sklearn_like_model.GAN.GAN_MixIn import GAN_loss_builder_MixIn
from script.model.sklearn_like_model.Mixin import Ys_MixIn, Xs_MixIn, zs_MixIn, noise_MixIn
from script.model.sklearn_like_model.BaseModel import BaseModel
from script.util.Stacker import Stacker
from script.util.tensor_ops import *
from script.util.summary_func import *
from pprint import pformat
import numpy as np


def param_to_dict(**kwargs):
    return kwargs


class basicAAEPropertyMixIn(Xs_MixIn, Ys_MixIn, zs_MixIn, noise_MixIn):
    train_ops_name = [
        'train_AE',
        'train_D_gauss',
        'train_G_gauss',
        'train_D_cate',
        'train_G_cate',
        'op_inc_global_step',
        # 'train_D_recon',
        # 'train_G_recon'
    ]
    loss_names = [
        'loss_AE',
        'loss_G_gauss',
        'loss_G_cate',
        'loss_D_gauss',
        'loss_D_cate',
        # 'loss_G_recon',
        # 'loss_D_recon'
    ]

    def __init__(self):
        Xs_MixIn.__init__(self)
        Ys_MixIn.__init__(self)
        zs_MixIn.__init__(self)
        noise_MixIn.__init__(self)

    @property
    def _code_ops(self):
        return getattr(self, 'latent_code')

    @property
    def _recon_ops(self):
        return getattr(self, 'Xs_recon')

    @property
    def _generate_ops(self):
        return getattr(self, 'Xs_gen')

    @property
    def _train_ops(self):
        return [getattr(self, name, None) for name in self.train_ops_name]

    @property
    def _metric_ops(self):
        return {name: getattr(self, name) for name in self.loss_names}

    @property
    def _predict_ops(self):
        return getattr(self, 'predict_index', None)

    @property
    def _score_ops(self):
        return getattr(self, 'acc_mean', None)

    @property
    def _predict_proba_ops(self):
        return getattr(self, 'hs', None)


def common_linear_stack(stack: Stacker, net_shapes, bn=True, activation='relu') -> Stacker:
    for shape in net_shapes:
        stack.linear(shape)
        if bn:
            stack.bn()

        stack.activation(activation)

    return stack


def AAE_encoder(Xs, net_shapes, z_size, Y_flatten_size,
                linear_stack_bn=False, linear_stack_activation='relu',
                tail_bn=False, tail_activation='none',
                reuse=False, name='encoder', ):
    def encoder_head(Xs):
        stack = Stacker(Xs)
        stack.flatten()
        return stack

    def encoder_tail(stack: Stacker, Y_flatten_size, z_size, bn=False, activation='none'):
        stack.linear(z_size + Y_flatten_size)
        if bn:
            stack.bn()
        stack.activation(activation)

        zs = stack.last_layer[:, :z_size]
        Ys_gen = stack.last_layer[:, z_size:]
        hs = softmax(Ys_gen)
        return zs, Ys_gen, hs

    with tf.variable_scope(name, reuse=reuse):
        stack = encoder_head(Xs)
        stack = common_linear_stack(stack, net_shapes, bn=linear_stack_bn, activation=linear_stack_activation)
        zs, Ys_gen, hs = encoder_tail(stack, Y_flatten_size, z_size, bn=tail_bn, activation=tail_activation)

    return zs, Ys_gen, hs


def AAE_decoder(zs, Ys, net_shapes, X_flatten_size, Xs_shape,
                linear_stack_bn=False, linear_stack_activation='relu',
                tail_bn=False, tail_activation='sigmoid',
                reuse=False, name='decoder'):
    def decoder_head(zs, Ys) -> Stacker:
        stack = Stacker(concat((zs, Ys), axis=1))
        return stack

    def decoder_tail(stack: Stacker, flatten_size, output_shape, bn=False, activation='sigmoid') -> Stacker:
        stack.linear(flatten_size)
        if bn:
            stack.bn()
        stack.activation(activation)
        stack.reshape(output_shape)

        return stack

    with tf.variable_scope(name, reuse=reuse):
        stack = decoder_head(zs, Ys)
        stack = common_linear_stack(stack, net_shapes, bn=linear_stack_bn, activation=linear_stack_activation)
        stack = decoder_tail(stack, X_flatten_size, Xs_shape, bn=tail_bn, activation=tail_activation)

    return stack.last_layer


def tf_sharpen_filter(Xs):
    return 0.5 * tf.tanh(Xs * 5 - 2.5) + 0.5


class AAE(BaseModel, basicAAEPropertyMixIn, GAN_loss_builder_MixIn):
    def _build_input_shapes(self, shapes):
        ret = {}
        ret.update(self._build_Xs_input_shape(shapes))

        ret.update(self._build_Ys_input_shape(shapes))

        shapes['zs'] = [None, self.z_size]
        ret.update(self._build_zs_input_shape(shapes))

        shapes['noise'] = [None] + list(ret['X_shape'])
        ret.update(self._build_noise_input_shape(shapes))

        return ret

    def __init__(self, batch_size=100, learning_rate=0.01, beta1=0.9, L1_norm_lambda=0.001, latent_code_size=32,
                 encoder_net_shapes=(512,), decoder_net_shapes=(512,), D_gauss_net_shapes=(512, 512),
                 D_cate_net_shapes=(512, 512), with_noise=False, noise_intensity=1.,
                 encoder_kwargs=None, decoder_kwargs=None,
                 verbose=10):
        BaseModel.__init__(self, verbose)
        basicAAEPropertyMixIn.__init__(self)
        GAN_loss_builder_MixIn.__init__(self)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.L1_norm_lambda = L1_norm_lambda
        self.latent_code_size = latent_code_size
        self.z_size = latent_code_size

        self.encoder_net_shapes = encoder_net_shapes
        self.decoder_net_shapes = decoder_net_shapes
        self.D_cate_net_shapes = D_cate_net_shapes
        self.D_gauss_net_shapes = D_gauss_net_shapes

        self.with_noise = with_noise
        self.noise_intensity = noise_intensity

        if encoder_kwargs is None:
            encoder_kwargs = {}
        self.encoder_kwargs = encoder_kwargs

        if decoder_kwargs is None:
            decoder_kwargs = {}
        self.decoder_kwargs = decoder_kwargs

    @staticmethod
    def discriminator(Xs, net_shapes, reuse=False, name='discriminator'):
        with tf.variable_scope(name, reuse=reuse):
            layer = Stacker(flatten(Xs))
            for shape in net_shapes:
                layer.linear_block(shape, relu)

            layer.linear(1)
            layer.sigmoid()

        return layer.last_layer

    def _build_main_graph(self):
        self.Xs = placeholder(tf.float32, self.Xs_shape, name='Xs')
        self.Ys = placeholder(tf.float32, self.Ys_shape, name='Ys')
        self.zs = placeholder(tf.float32, self.zs_shape, name='zs')
        self.noises = placeholder(tf.float32, self.noises_shape, name='noises')

        self.Xs_noised = tf.add(self.Xs, self.noises, name='Xs_noised')
        if self.with_noise:
            Xs = self.Xs_noised
        else:
            Xs = self.Xs

        self.zs_gen, self.Ys_gen, self.hs = AAE_encoder(Xs, self.encoder_net_shapes, self.z_size, self.Y_flatten_size)
        self.latent_code = self.zs_gen
        self.latent_code = tf_minmax_scaling(self.latent_code)

        self.Xs_recon = AAE_decoder(self.latent_code, self.Ys_gen, self.decoder_net_shapes, self.X_flatten_size,
                                    self.Xs_shape)
        self.Xs_recon = tf_sharpen_filter(self.Xs_recon)

        self.Xs_gen = AAE_decoder(self.zs, self.Ys, self.decoder_net_shapes, self.X_flatten_size, self.Xs_shape,
                                  reuse=True)
        self.Xs_gen = tf_sharpen_filter(self.Xs_gen)

        self.D_gauss_real = self.discriminator(
            self.zs, self.D_gauss_net_shapes, name='discriminator_gauss')
        self.D_gauss_gen = self.discriminator(
            self.zs_gen, self.D_gauss_net_shapes, reuse=True, name='discriminator_gauss')

        self.D_cate_real = self.discriminator(
            self.Ys, self.D_cate_net_shapes, name='discriminator_cate')
        self.D_cate_gen = self.discriminator(
            self.Ys_gen, self.D_cate_net_shapes, reuse=True, name='discriminator_cate')

        head = get_scope()
        self.vars_encoder = collect_vars(join_scope(head, 'encoder'))
        self.vars_decoder = collect_vars(join_scope(head, 'decoder'))
        self.vars_discriminator_gauss = collect_vars(join_scope(head, 'discriminator_gauss'))
        self.vars_discriminator_cate = collect_vars(join_scope(head, 'discriminator_cate'))

        # self.predict_index = tf.cast(tf.argmax(self.hs, 1), tf.float32, name="predict_index")
        # self.label_index = onehot_to_index(self.Ys)
        # self.acc = tf.cast(tf.equal(self.predict_index, self.label_index), tf.float64, name="acc")
        # self.acc_mean = tf.reduce_mean(self.acc, name="acc_mean")

    def _build_loss_ops(self):
        # AE loss
        self.loss_AE = tf.squared_difference(self.Xs, self.Xs_recon, name="loss_AE")
        self.loss_AE_mean = tf.reduce_sum(self.loss_AE, name="loss_AE_mean")

        # D gauss loss
        self.loss_D_gauss_real, self.loss_D_gauss_gen, self.loss_D_gauss, self.loss_G_gauss = \
            self._build_GAN_loss(self.D_gauss_real, self.D_gauss_gen, 'WGAN')

        self.loss_D_cate_real, self.loss_D_cate_gen, self.loss_D_cate, self.loss_G_cate = \
            self._build_GAN_loss(self.D_cate_real, self.D_cate_gen, 'WGAN')

        # # classifier phase
        # self.loss_clf = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Ys,
        #                                                            logits=self.hs,
        #                                                            name='loss_clf')
        # self.loss_clf_mean = tf.reduce_mean(self.loss_clf, name='loss_clf_mean')

    def _build_train_ops(self):
        # reconstruction phase
        self.train_AE = tf.train.AdamOptimizer(
            self.learning_rate,
            self.beta1
        ).minimize(
            self.loss_AE_mean,
            var_list=self.vars_decoder + self.vars_encoder
        )

        # regularization phase
        self.train_D_gauss = tf.train.AdamOptimizer(
            self.learning_rate,
            self.beta1
        ).minimize(
            self.loss_D_gauss,
            var_list=self.vars_discriminator_gauss
        )

        self.train_G_gauss = tf.train.AdamOptimizer(
            self.learning_rate,
            self.beta1
        ).minimize(
            self.loss_G_gauss,
            var_list=self.vars_encoder
        )

        self.train_D_cate = tf.train.AdamOptimizer(
            self.learning_rate * 0.2,
            self.beta1
        ).minimize(
            self.loss_D_cate,
            var_list=self.vars_discriminator_cate
        )

        self.train_G_cate = tf.train.AdamOptimizer(
            self.learning_rate,
            self.beta1
        ).minimize(
            self.loss_G_cate,
            var_list=self.vars_encoder
        )

        # self.train_clf = tf.train.AdamOptimizer(
        #     self.learning_rate,
        #     self.beta1
        # ).minimize(
        #     self.loss_clf,
        #     var_list=self.vars_encoder
        #
        # )

    def train(self, Xs, Ys, epoch=1, save_interval=None, batch_size=None, z_dist_kwargs=None):
        self._prepare_train(Xs=Xs, Ys=Ys)
        dataset = self.to_dummyDataset(Xs=Xs, Ys=Ys)

        if batch_size is None:
            batch_size = self.batch_size

        iter_num = 0
        iter_per_epoch = dataset.size // batch_size
        self.log.info("train epoch {}, iter/epoch {}".format(epoch, iter_per_epoch))

        for e in trange(epoch):
            dataset.shuffle()

            total = param_to_dict(
                loss_AE=[],
                loss_G_gauss=[],
                loss_G_cate=[],
                loss_D_cate=[],
                loss_D_gauss=[],
                # loss_clf=0
            )
            for i in range(iter_per_epoch):
                iter_num += 1

                Xs, Ys = dataset.next_batch(batch_size, batch_keys=['Xs', 'Ys'])
                # zs = self.get_zs_rand_normal(batch_size)
                zs = self.get_zs_rand_beta(batch_size)
                noise = self.get_noises(Xs.shape, self.noise_intensity)
                self.sess.run(self._train_ops,
                              feed_dict={
                                  self._Xs:     Xs,
                                  self._Ys:     Ys,
                                  self._zs:     zs,
                                  self._noises: noise
                              })

                metric = self.metric(Xs, Ys)
                total = {key: total[key] + [metric[key]] for key in total}

            total = {key: np.mean(val) for key, val in total.items()}
            pformat(total)
            self.log.info(total)

            if save_interval is not None and e % save_interval == 0:
                self.save()

    def code(self, Xs):
        return self.get_tf_values(self._code_ops, {
            self._Xs:     Xs,
            self._noises: (self.get_noises(Xs.shape))
        })

    def code_Y(self, Xs):
        return self.get_tf_values(self.Ys_gen, {
            self._Xs:     Xs,
            self._noises: (self.get_noises(Xs.shape))
        })

    @staticmethod
    def sharpen_filter(Xs, sharpness=5):
        sharpen = 0.5 * np.tanh(Xs * sharpness - sharpness / 2) + 0.5
        return sharpen

    def recon_sharpen(self, Xs, Ys, sharpness=5):
        recon = self.get_tf_values(self._recon_ops, {
            self._Xs:     Xs,
            self._Ys:     Ys,
            self._noises: (self.get_noises(Xs.shape))
        })
        recon_sharpen = self.sharpen_filter(recon, sharpness=sharpness)
        return recon_sharpen

    def recon(self, Xs, Ys):
        recon = self.get_tf_values(self._recon_ops, {
            self._Xs:     Xs,
            self._Ys:     Ys,
            self._noises: (self.get_noises(Xs.shape))
        })
        return recon

    def generate(self, Ys, zs=None):
        if zs is None:
            size = len(Ys)
            zs = self.get_zs_rand_beta(size)
        return self.get_tf_values(self._generate_ops, {
            self._zs: zs,
            self._Ys: Ys
        })

    def metric(self, Xs, Ys, mean=True):
        metric = self.get_tf_values(
            self._metric_ops,
            {
                self._Xs:     np.array(Xs),
                self._Ys:     Ys,
                self._zs:     (self.get_z_rand_uniform([Xs.shape[0], self.z_size])),
                self._noises: (self.get_noises(Xs.shape))
            },
            wrap_dict=True
        )

        if mean:
            metric = {key: np.mean(val) for key, val in metric.items()}
        return metric

    def predict_proba(self, Xs):
        return self.get_tf_values(self._predict_ops, {
            self._Xs:     Xs,
            self._noises: (self.get_noises(Xs.shape))
        })

    def predict(self, Xs):
        return self.get_tf_values(self._predict_ops, {
            self._Xs:     Xs,
            self._noises: (self.get_noises(Xs.shape))
        })

    def score(self, Xs, Ys):
        return self.get_tf_values(
            self._score_ops,
            {
                self._Xs:     np.array(Xs),
                self._Ys:     Ys,
                self._zs:     self.get_z_rand_uniform([Xs.shape[0], self.z_size]),
                self._noises: self.get_noises(Xs.shape)
            }
        )

    def augmentation(self, Xs, Ys, code_gap=None):
        if code_gap is None:
            size = len(Xs)
            code_gap = self.get_zs_rand_uniform(size)

        code = self.code(Xs) + code_gap
        code_Y = self.code_Y(Xs)
        augmentation = self.generate(code_Y, code)
        return augmentation
