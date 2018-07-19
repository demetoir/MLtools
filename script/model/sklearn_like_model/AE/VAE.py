from script.model.sklearn_like_model.AE.AE import AE
from script.util.Stacker import Stacker
from script.util.tensor_ops import *
from script.util.summary_func import *


def tf_minmax_scaling(x, epsilon=1e-7):
    min_ = tf.reduce_min(x)
    max_ = tf.reduce_max(x)
    return (x - min_) / (max_ - min_ + epsilon)


def tf_z_score_normalize(x: tf.Tensor):
    if len(x.shape) is not 1:
        raise TypeError('x rank must be 1')
    mean, stddev = tf.nn.moments(x, 0)
    return (x - mean) / stddev


def common_linear_stack(stack: Stacker, net_shapes, bn=True, activation='relu') -> Stacker:
    for shape in net_shapes:
        stack.linear(shape)
        if bn:
            stack.bn()

        stack.activation(activation)
        stack.lrelu()
    return stack


def encoder_head(Xs):
    stack = Stacker(Xs)
    stack.flatten()
    return stack


def encoder_tail(stack: Stacker, latent_code_size, bn=False, activation='none') -> Stacker:
    stack.linear(latent_code_size * 2)
    if bn:
        stack.bn()
    stack.activation(activation)
    return stack


def basicVAE_Encoder(
        Xs, net_shapes, latent_code_size,
        linear_stack_bn=False, linear_stack_activation='relu',
        tail_bn=False, tail_activation='none',
        reuse=False, name='encoder'):
    with tf.variable_scope(name, reuse=reuse):
        stack = encoder_head(Xs)
        stack = common_linear_stack(stack, net_shapes, bn=linear_stack_bn, activation=linear_stack_activation)
        stack = encoder_tail(stack, latent_code_size, bn=tail_bn, activation=tail_activation)
    return stack.last_layer


def decoder_head(latents) -> Stacker:
    stack = Stacker(latents)
    return stack


def decoder_tail(stack: Stacker, flatten_size, output_shape, bn=False, activation='sigmoid') -> Stacker:
    stack.linear(flatten_size)
    if bn:
        stack.bn()
    stack.activation(activation)
    stack.reshape(output_shape)
    return stack


def basicVAE_Decoder(latents, net_shapes, flatten_size, output_shape,
                     linear_stack_bn=False, linear_stack_activation='relu',
                     tail_bn=False, tail_activation='sigmoid',

                     reuse=False, name='decoder'):
    with tf.variable_scope(name, reuse=reuse):
        stack = decoder_head(latents)
        stack = common_linear_stack(stack, net_shapes, bn=linear_stack_bn, activation=linear_stack_activation)
        stack = decoder_tail(stack, flatten_size, output_shape, bn=tail_bn, activation=tail_activation)

    return stack.last_layer


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

    @property
    def loss_names(self):
        return self._loss_builder_funcs.keys()

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
        return loss

    def MSE_with_KL_loss(self, X, X_out, mean, std, KL_D_rate):
        regularization_error = self._regularization_error(mean, std, KL_D_rate)
        MSE = self._MSE(X, X_out)

        loss = MSE + regularization_error
        return loss

    def RMSE_with_KL_loss(self, X, X_out, mean, std, KL_D_rate):
        RMSE = self._RMSE(X, X_out)
        regularization_error = self._regularization_error(mean, std, KL_D_rate)

        loss = RMSE + regularization_error
        return loss

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


class VAE(AE, VAE_loss_builder_MixIn):
    def __init__(self, batch_size=100, learning_rate=0.01, beta1=0.5, L1_norm_lambda=0.001, latent_code_size=32,
                 encoder_net_shapes=(512,), decoder_net_shapes=(512,), with_noise=False, noise_intensity=1.,
                 loss_type='VAE', encoder_kwargs=None, decoder_kwargs=None,
                 KL_D_rate=0.01, verbose=10):
        AE.__init__(
            self, batch_size, learning_rate, beta1, L1_norm_lambda, latent_code_size,
            encoder_net_shapes, decoder_net_shapes, with_noise, noise_intensity,
            encoder_kwargs, decoder_kwargs, verbose)
        VAE_loss_builder_MixIn.__init__(self)

        self.KL_D_rate = KL_D_rate
        self.loss_type = loss_type

    def _build_main_graph(self):
        self.Xs = placeholder(tf.float32, self.Xs_shape, name='Xs')
        self.zs = placeholder(tf.float32, self.zs_shape, name='zs')
        self.noises = placeholder(tf.float32, self.noises_shape, name='noises')

        self.Xs_noised = tf.add(self.Xs, self.noises, name='Xs_noised')
        if self.with_noise:
            Xs = self.Xs_noised
        else:
            Xs = self.Xs

        self.h = basicVAE_Encoder(Xs, self.encoder_net_shapes, self.latent_code_size, **self.encoder_kwargs)
        self.h = tf.identity(self.h, 'h')

        self.mean = tf.identity(self.h[:, :self.latent_code_size], 'mean')
        self.std = tf.identity(tf.nn.softplus(self.h[:, self.latent_code_size:]), 'std')
        # self.mean = tf.nn.tanh(self.mean)
        # self.std = tf.nn.leaky_relu(self.std)

        self.latent_code = self.mean + self.std * tf.random_normal(tf.shape(self.mean), 0, 1, dtype=tf.float32)

        self.latent_code = tf_minmax_scaling(self.latent_code)

        self.Xs_recon = basicVAE_Decoder(
            self.latent_code, self.decoder_net_shapes, self.X_flatten_size, self.Xs_shape,
            **self.decoder_kwargs
        )
        self.Xs_gen = basicVAE_Decoder(
            self.zs, self.decoder_net_shapes, self.X_flatten_size, self.Xs_shape, reuse=True,
            **self.decoder_kwargs
        )

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
