from tqdm import trange
from script.model.sklearn_like_model.Mixin import Ys_MixIn, Xs_MixIn, zs_MixIn, noise_MixIn
from script.model.sklearn_like_model.BaseModel import BaseModel
from script.util.Stacker import Stacker
from script.util.tensor_ops import *
from script.util.summary_func import *
import numpy as np


class basicAAEPropertyMixIn(Xs_MixIn, Ys_MixIn, zs_MixIn, noise_MixIn):
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
        return [
            getattr(self, 'train_AE', None),
            getattr(self, 'train_D_gauss', None),
            getattr(self, 'train_D_cate', None),
            getattr(self, 'train_G_gauss', None),
            getattr(self, 'train_G_cate', None),
            getattr(self, 'train_clf', None),
            getattr(self, 'op_inc_global_step', None),
        ]

    @property
    def _metric_ops(self):
        return [
            getattr(self, 'loss_AE'),
            getattr(self, 'loss_G_gauss'),
            getattr(self, 'loss_G_cate'),
            getattr(self, 'loss_D_gauss'),
            getattr(self, 'loss_D_cate'),
            getattr(self, 'loss_clf'),
        ]

    @property
    def _predict_ops(self):
        return getattr(self, 'predict_index', None)

    @property
    def _score_ops(self):
        return getattr(self, 'acc_mean', None)

    @property
    def _predict_proba_ops(self):
        return getattr(self, 'hs', None)


class AAE(BaseModel, basicAAEPropertyMixIn):
    _params_keys = [
        'batch_size',
        'learning_rate',
        'beta1',
        'L1_norm_lambda',
        'K_average_top_k_loss',
        'code_size',
        'z_size',
        'encoder_net_shapes',
        'decoder_net_shapes',
        'D_gauss_net_shapes',
        'with_noise',
        'noise_intensity',
    ]

    def __init__(self, batch_size=100, learning_rate=0.01, beta1=0.5, L1_norm_lambda=0.001, latent_code_size=32,
                 z_size=32, encoder_net_shapes=None, decoder_net_shapes=None, D_gauss_net_shapes=None,
                 D_cate_net_shapes=None, with_noise=False, noise_intensity=1.,
                 verbose=10):
        BaseModel.__init__(self, verbose)
        basicAAEPropertyMixIn.__init__(self)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.L1_norm_lambda = L1_norm_lambda
        self.latent_code_size = latent_code_size
        self.z_size = z_size

        if encoder_net_shapes is None:
            self.encoder_net_shapes = [512, 256, 128]
        else:
            self.encoder_net_shapes = decoder_net_shapes
        if decoder_net_shapes is None:
            self.decoder_net_shapes = [128, 256, 512]
        else:
            self.decoder_net_shapes = decoder_net_shapes
        if D_cate_net_shapes is None:
            self.D_cate_net_shapes = [512, 512]
        else:
            self.D_cate_net_shapes = D_cate_net_shapes
        if D_gauss_net_shapes is None:
            self.D_gauss_net_shapes = [512, 512]
        else:
            self.D_gauss_net_shapes = D_cate_net_shapes

        self.with_noise = with_noise
        self.noise_intensity = noise_intensity

    def _build_input_shapes(self, shapes):
        ret = {}
        ret.update(self._build_Xs_input_shape(shapes))

        ret.update(self._build_Ys_input_shape(shapes))

        shapes['zs'] = [None, self.z_size]
        ret.update(self._build_zs_input_shape(shapes))

        shapes['noise'] = [None] + list(ret['X_shape'])
        ret.update(self._build_noise_input_shape(shapes))

        return ret

    def encoder(self, Xs, net_shapes, reuse=False, name='encoder'):
        with tf.variable_scope(name, reuse=reuse):
            stack = Stacker(Xs)
            stack.flatten()

            for shape in net_shapes:
                stack.linear_block(shape, relu)

            stack.linear_block(self.z_size + self.Y_flatten_size, relu)
            zs = stack.last_layer[:, :self.z_size]
            Ys_gen = stack.last_layer[:, self.z_size:]

            hs = softmax(Ys_gen)
        return zs, Ys_gen, hs

    def decoder(self, zs, Ys, net_shapes, reuse=False, name='decoder'):
        with tf.variable_scope(name, reuse=reuse):
            stack = Stacker(concat((zs, Ys), axis=1))

            for shape in net_shapes:
                stack.linear_block(shape, relu)

            stack.linear_block(self.X_flatten_size, sigmoid)
            stack.reshape(self.Xs_shape)

        return stack.last_layer

    @staticmethod
    def discriminator_gauss(zs, net_shapes, reuse=False, name='discriminator_gauss'):
        with tf.variable_scope(name, reuse=reuse):
            layer = Stacker(zs)
            for shape in net_shapes:
                layer.linear_block(shape, relu)

            layer.linear(1)
            layer.sigmoid()

        return layer.last_layer

    @staticmethod
    def discriminator_cate(zs, net_shapes, reuse=False, name='discriminator_cate'):
        with tf.variable_scope(name, reuse=reuse):
            layer = Stacker(zs)
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

        self.zs_gen, self.Ys_gen, self.hs = self.encoder(Xs, self.encoder_net_shapes)
        self.latent_code = self.zs_gen
        self.Xs_recon = self.decoder(self.zs_gen, self.Ys_gen, self.decoder_net_shapes)
        self.Xs_gen = self.decoder(self.zs, self.Ys, self.decoder_net_shapes, reuse=True)

        self.D_gauss_real = self.discriminator_gauss(self.zs, self.D_gauss_net_shapes)
        self.D_gauss_gen = self.discriminator_gauss(self.zs_gen, self.D_gauss_net_shapes, reuse=True)

        self.D_cate_real = self.discriminator_cate(self.Ys, self.D_cate_net_shapes)
        self.D_cate_gen = self.discriminator_cate(self.Ys_gen, self.D_cate_net_shapes, reuse=True)

        head = get_scope()
        self.vars_encoder = collect_vars(join_scope(head, 'encoder'))
        self.vars_decoder = collect_vars(join_scope(head, 'decoder'))
        self.vars_discriminator_gauss = collect_vars(join_scope(head, 'discriminator_gauss'))
        self.vars_discriminator_cate = collect_vars(join_scope(head, 'discriminator_cate'))

        # self.predict = tf.equal(tf.argmax(self.hs, 1), tf.argmax(self.Ys, 1), name='predict')

        self.predict_index = tf.cast(tf.argmax(self.hs, 1), tf.float32, name="predict_index")
        self.label_index = onehot_to_index(self.Ys)
        self.acc = tf.cast(tf.equal(self.predict_index, self.label_index), tf.float64, name="acc")
        self.acc_mean = tf.reduce_mean(self.acc, name="acc_mean")

    def _build_loss_function(self):
        # AE loss
        self.loss_AE = tf.squared_difference(self.Xs, self.Xs_recon, name="loss_AE")
        self.loss_AE_mean = tf.reduce_sum(self.loss_AE, name="loss_AE_mean")

        # D gauss loss
        self.loss_D_gauss_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_gauss_real),
                                                                         logits=self.D_gauss_real,
                                                                         name='loss_D_gauss_real')
        self.loss_D_gauss_gen = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.D_gauss_gen),
                                                                        logits=self.D_gauss_gen,
                                                                        name='loss_D_gauss_gen')

        self.loss_D_gauss = tf.add(self.loss_D_gauss_real, self.loss_D_gauss_gen, name='loss_D_gauss')
        self.loss_D_gauss_mean = tf.reduce_mean(self.loss_D_gauss, name='loss_D_gauss_mean')

        # D cate loss
        self.loss_D_cate_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_cate_real),
                                                                        logits=self.D_cate_real,
                                                                        name='loss_D_cate_real')
        self.loss_D_cate_gen = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.D_cate_gen),
                                                                       logits=self.D_cate_gen,
                                                                       name='loss_D_cate_gen')
        self.loss_D_cate = tf.add(self.loss_D_cate_real, self.D_cate_gen, name='loss_D_cate')
        self.loss_D_cate_mean = tf.reduce_mean(self.loss_D_cate, name='loss_D_cate_mean')

        # G gauss loss
        self.loss_G_gauss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_gauss_gen),
                                                                    logits=self.D_gauss_gen,
                                                                    name='loss_G_gauss')
        self.loss_G_gauss_mean = tf.reduce_mean(self.loss_G_gauss, name='loss_G_gauss_mean')

        # G cate loss
        self.loss_G_cate = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_cate_gen),
                                                                   logits=self.D_cate_gen,
                                                                   name='loss_G_cate')
        self.loss_G_cate_mean = tf.reduce_mean(self.loss_G_cate, name='loss_G_cate_mean')

        # classifier phase
        self.loss_clf = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Ys,
                                                                   logits=self.hs,
                                                                   name='loss_clf')
        self.loss_clf_mean = tf.reduce_mean(self.loss_clf, name='loss_clf_mean')

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

        self.train_D_cate = tf.train.AdamOptimizer(
            self.learning_rate,
            self.beta1
        ).minimize(
            self.loss_D_cate,
            var_list=self.vars_discriminator_cate
        )

        self.train_G_gauss = tf.train.AdamOptimizer(
            self.learning_rate,
            self.beta1
        ).minimize(
            self.loss_G_gauss,
            var_list=self.vars_encoder
        )

        self.train_G_cate = tf.train.AdamOptimizer(
            self.learning_rate,
            self.beta1
        ).minimize(
            self.loss_G_cate,
            var_list=self.vars_encoder
        )

        self.train_clf = tf.train.AdamOptimizer(
            self.learning_rate,
            self.beta1
        ).minimize(
            self.loss_clf,
            var_list=self.vars_encoder

        )

    def random_z(self):
        pass

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
            total_AE = 0
            total_G_gauss = 0
            total_G_cate = 0
            total_D_gauss = 0
            total_D_cate = 0
            total_clf = 0
            for i in range(iter_per_epoch):
                iter_num += 1

                Xs, Ys = dataset.next_batch(batch_size, batch_keys=['Xs', 'Ys'])
                # print([batch_size, self.z_size])
                zs = self.get_z_rand_uniform([batch_size, self.z_size])
                noise = self.get_noises(Xs.shape, self.noise_intensity)
                self.sess.run(self._train_ops,
                              feed_dict={self._Xs: Xs, self._Ys: Ys, self._zs: zs, self._noises: noise})

                loss = self.sess.run(self._metric_ops,
                                     feed_dict={self._Xs: Xs, self._Ys: Ys, self._zs: zs, self._noises: noise})

                loss_AE, loss_G_gauss, loss_G_cate, loss_D_gauss, loss_D_cate, loss_clf = loss
                total_AE += np.sum(loss_AE) / dataset.size
                total_G_gauss += np.sum(loss_G_gauss) / dataset.size
                total_G_cate += np.sum(loss_G_cate) / dataset.size
                total_D_gauss += np.sum(loss_D_gauss) / dataset.size
                total_D_cate += np.sum(loss_D_cate) / dataset.size
                total_clf += np.sum(loss_clf) / dataset.size

            self.log.info(
                "e:{} loss AE={}, G_gauss={}, G_cate={}, D_gauss={}, D_cate={}, "
                "clf={}".format(e, total_AE, total_G_gauss, total_G_cate, total_D_gauss, total_D_cate, total_clf))
            # Xs, Ys = dataset.next_batch(batch_size, batch_keys=['Xs', 'Ys'], look_up=False)
            # zs = self.get_z_noise([batch_size, self.z_size])
            # loss = self.sess.run(self._metric_ops, feed_dict={self._Xs: Xs, self._Ys: Ys, self._zs: zs})

            if save_interval is not None and e % save_interval == 0:
                self.save()

    def code(self, Xs):
        noise = self.get_noises(Xs.shape)
        return self.get_tf_values(self._code_ops, {self._Xs: Xs, self._noises: noise})

    def recon(self, Xs, Ys):
        noise = self.get_noises(Xs.shape)
        return self.get_tf_values(self._recon_ops, {self._Xs: Xs, self._Ys: Ys, self._noises: noise})

    def generate(self, zs, Ys):
        return self.get_tf_values(self._recon_ops, {self._zs: zs, self._Ys: Ys})

    def metric(self, Xs, Ys):
        Xs = np.array(Xs)
        zs = self.get_z_rand_uniform([Xs.shape[0], self.z_size])
        noises = self.get_noises(Xs.shape)
        return self.get_tf_values(self._metric_ops, {self._Xs: Xs, self._Ys: Ys, self._zs: zs, self._noises: noises})

    def predict_proba(self, Xs):
        noises = self.get_noises(Xs.shape)
        return self.get_tf_values(self._predict_ops, {self._Xs: Xs, self._noises: noises})

    def predict(self, Xs):
        noises = self.get_noises(Xs.shape)
        return self.get_tf_values(self._predict_ops, {self._Xs: Xs, self._noises: noises})

    def score(self, Xs, Ys):
        Xs = np.array(Xs)
        zs = self.get_z_rand_uniform([Xs.shape[0], self.z_size])
        noises = self.get_noises(Xs.shape)
        return self.get_tf_values(self._score_ops, {self._Xs: Xs, self._Ys: Ys, self._zs: zs, self._noises: noises})
