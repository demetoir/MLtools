import numpy as np
from tqdm import trange
from script.model.sklearn_like_model.AE.AE import basicAEPropertyMixIn
from script.model.sklearn_like_model.BaseModel import BaseModel
from script.model.sklearn_like_model.Mixin import Ys_MixIn
from script.util.Stacker import Stacker
from script.util.tensor_ops import *


class CVAE_MixIn(basicAEPropertyMixIn, Ys_MixIn):

    def __init__(self):
        basicAEPropertyMixIn.__init__(self)
        Ys_MixIn.__init__(self)


class VAE_loss_builder_MixIn:

    def __init__(self):
        cls = self.__class__
        self._loss_builder_funcs = {
            'VAE': cls.VAE_loss,
            'MSE_with_KL': cls.MSE_with_KL_loss,
            'RMSE_with_KL': cls.RMSE_with_KL_loss,
            'recon_only': cls.recon_only_loss,
            'MSE_only': cls.MSE_only_loss,
            'RMSE_only': cls.RMSE_only_loss,
        }

    @staticmethod
    def _recon_error(X, X_out):
        cross_entropy = tf.reduce_sum(X * tf.log(X_out) + (1 - X) * tf.log(1 - X_out), axis=1)
        recon_error = -1 * cross_entropy
        return recon_error

    @staticmethod
    def _regularization_error(mean, std, KL_D_rate):
        KL_Divergence = 0.5 * tf.reduce_sum(
            1 - tf.log(tf.square(std) + 1e-8) + tf.square(mean) + tf.square(std), axis=1)

        regularization_error = KL_Divergence * KL_D_rate
        return regularization_error

    @staticmethod
    def _MSE(X, X_out):
        MSE = tf.reduce_sum(tf.squared_difference(X, X_out), axis=1)
        return MSE

    @staticmethod
    def _RMSE(X, X_out):
        RMSE = tf.sqrt(tf.reduce_sum(tf.squared_difference(X, X_out), axis=1))
        return RMSE

    def VAE_loss(self, X, X_out, mean, std, KL_D_rate):
        # in autoencoder's perspective loss can be divide to reconstruct error and regularization error
        recon_error = self._recon_error(X, X_out)
        regularization_error = self._regularization_error(mean, std, KL_D_rate)

        loss = recon_error + regularization_error
        return loss, recon_error, regularization_error

    def MSE_with_KL_loss(self, X, X_out, mean, std, KL_D_rate):
        regularization_error = self._regularization_error(mean, std, KL_D_rate)
        MSE = self._MSE(X, X_out)

        loss = MSE + regularization_error
        return loss, MSE, regularization_error

    def RMSE_with_KL_loss(self, X, X_out, mean, std, KL_D_rate):
        RMSE = self._RMSE(X, X_out)
        regularization_error = self._regularization_error(mean, std, KL_D_rate)

        loss = RMSE + regularization_error
        return loss, RMSE, regularization_error

    def RMSE_only_loss(self, X, X_out, mean, std, KL_D_rate):
        RMSE = self._RMSE(X, X_out)

        return RMSE

    def recon_only_loss(self, X, X_out, mean, std, KL_D_rate):
        recon_error = self._recon_error(X, X_out)

        return recon_error

    def MSE_only_loss(self, X, X_out, mean, std, KL_D_rate):
        MSE = self._MSE(X, X_out)

        return MSE

    def _build_VAE_loss(self, X, X_out, mean, std, KL_D_rate, loss_type='VAE'):
        return self._loss_builder_funcs[loss_type](self, X, X_out, mean, std, KL_D_rate)


class CVAE(BaseModel, CVAE_MixIn, VAE_loss_builder_MixIn):
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
        'with_noise',
        'noise_intensity',
        'loss_type',
        'KL_D_rate'
    ]

    def __init__(self, batch_size=100, learning_rate=0.01, beta1=0.5, L1_norm_lambda=0.001, latent_code_size=32,
                 encoder_net_shapes=(512,), decoder_net_shapes=(512,), with_noise=False, noise_intensity=1.,
                 loss_type='VAE', KL_D_rate=1.0, verbose=10):
        BaseModel.__init__(self, verbose)
        CVAE_MixIn.__init__(self)
        VAE_loss_builder_MixIn.__init__(self)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.L1_norm_lambda = L1_norm_lambda
        self.latent_code_size = latent_code_size
        self.z_size = self.latent_code_size
        self.loss_type = loss_type
        self.encoder_net_shapes = encoder_net_shapes
        self.decoder_net_shapes = decoder_net_shapes
        self.with_noise = with_noise
        self.noise_intensity = noise_intensity
        self.KL_D_rate = KL_D_rate

    def _build_input_shapes(self, shapes):
        ret = {}
        ret.update(self._build_Xs_input_shape(shapes))

        ret.update(self._build_Ys_input_shape(shapes))

        shapes['zs'] = [None, self.z_size]
        ret.update(self._build_zs_input_shape(shapes))

        shapes['noise'] = [None] + list(ret['X_shape'])
        ret.update(self._build_noise_input_shape(shapes))

        return ret

    def encoder(self, Xs, Ys, net_shapes, reuse=False, name='encoder'):
        with tf.variable_scope(name, reuse=reuse):
            stack = Stacker(concat((flatten(Xs), Ys), axis=1))

            for shape in net_shapes:
                stack.linear_block(shape, relu)

            stack.linear_block(self.z_size * 2, relu)

        return stack.last_layer

    def decoder(self, zs, Ys, net_shapes, reuse=False, name='decoder'):
        with tf.variable_scope(name, reuse=reuse):
            stack = Stacker(concat((zs, Ys), axis=1))

            for shape in net_shapes:
                stack.linear_block(shape, relu)

            stack.linear_block(self.X_flatten_size, sigmoid)
            stack.reshape(self.Xs_shape)

        return stack.last_layer

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

        self.h = self.encoder(Xs, self.Ys, self.encoder_net_shapes)

        self.mean = self.h[:, :self.z_size]
        self.std = tf.nn.softplus(self.h[:, self.z_size:])
        self.latent_code = self.mean + self.std * tf.random_normal(tf.shape(self.mean), 0, 1, dtype=tf.float32)

        self.Xs_recon = self.decoder(self.latent_code, self.Ys, self.decoder_net_shapes)
        self.Xs_gen = self.decoder(self.zs, self.Ys, self.decoder_net_shapes, reuse=True)

        head = get_scope()
        self.vars = collect_vars(join_scope(head, 'encoder'))
        self.vars += collect_vars(join_scope(head, 'decoder'))

    def _build_loss_function(self):
        X = flatten(self.Xs)
        X_out = flatten(self.Xs_recon)
        mean = self.mean
        std = self.std
        self.loss = self._build_VAE_loss(X, X_out, std, mean, self.KL_D_rate, self.loss_type)

        self.loss_mean = tf.reduce_mean(self.loss, name='loss_mean')

    def _build_train_ops(self):
        self.train_op = tf.train.AdamOptimizer(self.learning_rate, self.beta1).minimize(loss=self.loss,
                                                                                        var_list=self.vars)

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

            for i in range(iter_per_epoch):
                iter_num += 1

                Xs, Ys = dataset.next_batch(batch_size)
                noise = self.get_noises(Xs.shape, self.noise_intensity)
                self.sess.run(self._train_ops, feed_dict={self._Xs: Xs, self._Ys: Ys, self._noises: noise})

            Xs, Ys = dataset.next_batch(batch_size, look_up=False)
            noise = self.get_noises(Xs.shape)
            loss = self.sess.run(self._metric_ops, feed_dict={self._Xs: Xs, self._Ys: Ys, self._noises: noise})
            self.log.info("e:{e} loss : {loss}".format(e=e, loss=np.mean(loss)))

            if save_interval is not None and e % save_interval == 0:
                self.save()

    def code(self, Xs, Ys):
        noise = self.get_noises(Xs.shape)
        return self.get_tf_values(self._code_ops, {self._Xs: Xs, self.Ys: Ys, self._noises: noise})

    def recon(self, Xs, Ys):
        noise = self.get_noises(Xs.shape)
        return self.get_tf_values(self._recon_ops, {self._Xs: Xs, self.Ys: Ys, self._noises: noise})

    def metric(self, Xs, Ys):
        noise = self.get_noises(Xs.shape)
        return self.get_tf_values(self._metric_ops, {self._Xs: Xs, self.Ys: Ys, self._noises: noise})

    def generate(self, Ys, zs=None):
        if zs is None:
            zs = self.get_zs_rand_normal(len(Ys))
        return self.get_tf_values(self._generate_ops, {self._Ys: Ys, self._zs: zs})
