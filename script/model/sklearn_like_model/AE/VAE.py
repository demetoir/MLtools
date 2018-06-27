from script.model.sklearn_like_model.AE.AE import basicAE_DecoderMixIn
from script.model.sklearn_like_model.AE.BaseAutoEncoder import BaseAutoEncoder
from script.util.Stacker import Stacker
from script.util.tensor_ops import *
from script.util.summary_func import *


class basicVAE_EncoderMixIn:
    @staticmethod
    def encoder(Xs, net_shapes, latent_code_size, reuse=False, name='encoder'):
        with tf.variable_scope(name, reuse=reuse):
            stack = Stacker(Xs)
            stack.flatten()
            for shape in net_shapes:
                stack.linear_block(shape, relu)

            stack.linear_block(latent_code_size * 2, relu)

        return stack.last_layer


class VAE(BaseAutoEncoder, basicAE_DecoderMixIn, basicVAE_EncoderMixIn):
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
    ]
    _input_shape_keys = [
        'X_shape',
        'Xs_shape',
        'X_flatten_size',
        'z_shape',
        'zs_shape'
    ]

    def __init__(self, verbose=10, **kwargs):
        super(VAE, self).__init__(verbose, **kwargs)

        self.batch_size = 100
        self.learning_rate = 0.01
        self.beta1 = 0.5
        self.L1_norm_lambda = 0.001
        self.K_average_top_k_loss = 10
        self.latent_code_size = 32
        self.z_size = 32
        self.encoder_net_shapes = [512, 256, 128]
        self.decoder_net_shapes = [128, 256, 512]

        self.X_shape = None
        self.Xs_shape = None
        self.X_flatten_size = None
        self.z_shape = None
        self.zs_shape = None

    def build_input_shapes(self, shapes):
        X_shape = shapes['Xs']
        Xs_shape = [None] + list(X_shape)
        X_flatten_size = self.flatten_shape(X_shape)

        z_shape = [self.z_size]
        zs_shape = [None, self.z_size]

        ret = {
            'X_shape': X_shape,
            'Xs_shape': Xs_shape,
            'X_flatten_size': X_flatten_size,
            'z_shape': z_shape,
            'zs_shape': zs_shape
        }
        return ret

    def _build_main_graph(self):
        self.Xs = placeholder(tf.float32, self.Xs_shape, name='Xs')
        self.zs = placeholder(tf.float32, self.zs_shape, name='zs')

        self.h = self.encoder(self.Xs, self.encoder_net_shapes, self.latent_code_size)
        self.h = tf.identity(self.h, 'h')

        self.mean = tf.identity(self.h[:, :self.z_size], 'mean')
        self.std = tf.identity(tf.nn.softplus(self.h[:, self.z_size:]), 'std')
        self.latent_code = self.mean + self.std * tf.random_normal(tf.shape(self.mean), 0, 1, dtype=tf.float32)

        self.Xs_recon = self.decoder(self.latent_code, self.decoder_net_shapes, self.X_flatten_size, self.Xs_shape)
        self.Xs_gen = self.decoder(self.zs, self.decoder_net_shapes, self.X_flatten_size, self.Xs_shape, reuse=True)

        head = get_scope()
        self.vars = collect_vars(join_scope(head, 'encoder'))
        self.vars += collect_vars(join_scope(head, 'decoder'))

    def _build_loss_function(self):
        X = flatten(self.Xs)
        X_out = flatten(self.Xs_recon)
        mean = self.mean
        std = self.std

        self.cross_entropy = tf.reduce_sum(X * tf.log(X_out) + (1 - X) * tf.log(1 - X_out), axis=1)
        self.KL_Divergence = 0.5 * tf.reduce_sum(
            1 - tf.log(tf.square(std) + 1e-8) + tf.square(mean) + tf.square(std), axis=1)

        # in autoencoder's perspective loss can be divide to reconstruct error and regularization error
        # self.recon_error = -1 * self.cross_entropy
        # self.regularization_error = self.KL_Divergence
        # self.loss = self.recon_error + self.regularization_error

        # only cross entropy loss also work
        # self.loss = -1 * self.cross_entropy

        # using MSE than cross entropy loss also work but slow
        # self.MSE= tf.reduce_sum(tf.squared_difference(X, X_out), axis=1)
        # self.loss = self.MSE + self.KL_Divergence

        # this one also work
        # self.loss = self.MSE

        self.loss = tf.add(-1 * self.cross_entropy, self.KL_Divergence, name='loss')
        self.loss_mean = tf.reduce_mean(self.loss, name='loss_mean')

    def _build_train_ops(self):
        self.train_op = tf.train.AdamOptimizer(self.learning_rate, self.beta1).minimize(loss=self.loss,
                                                                                        var_list=self.vars)

    def random_z(self):
        pass
