from script.model.sklearn_like_model.BaseModel import BaseModel
from script.model.sklearn_like_model.AE.AE import basicAEPropertyMixIn
from script.util.Stacker import Stacker
from script.util.tensor_ops import *
from script.util.summary_func import *
from tqdm import trange
import numpy as np


class CAEPropertyMixIn:

    @property
    def _Ys(self):
        return getattr(self, 'Ys', None)


class CVAE(BaseModel, basicAEPropertyMixIn, CAEPropertyMixIn):
    _input_shape_keys = [
        'X_shape',
        'Xs_shape',
        'X_flatten_size',
        'z_shape',
        'zs_shape',
        'Y_shape',
        'Ys_shape',
        'noise_shape'
    ]
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
    ]

    def __init__(self, batch_size=100, learning_rate=0.01, beta1=0.5, L1_norm_lambda=0.001, latent_code_size=32,
                 z_size=32, encoder_net_shapes=None, decoder_net_shapes=None, with_noise=False, noise_intensity=1.,
                 verbose=10):
        super(CVAE, self).__init__(verbose)
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

        self.with_noise = with_noise
        self.noise_intensity = noise_intensity

        self.X_shape = None
        self.Xs_shape = None
        self.X_flatten_size = None

        self.Y_shape = None
        self.Ys_shape = None

        self.z_shape = None
        self.zs_shape = None

        self.noise_shape = None

    def _build_input_shapes(self, shapes):
        X_shape = shapes['Xs']
        Xs_shape = [None] + list(X_shape)
        X_flatten_size = self.flatten_shape(X_shape)

        z_shape = [self.z_size]
        zs_shape = [None, self.z_size]

        Y_shape = shapes['Ys']
        Ys_shape = [None] + list(Y_shape)

        noise_shape = [None] + list(X_shape)

        ret = {
            'X_shape': X_shape,
            'Xs_shape': Xs_shape,
            'X_flatten_size': X_flatten_size,
            'z_shape': z_shape,
            'zs_shape': zs_shape,
            'Y_shape': Y_shape,
            'Ys_shape': Ys_shape,
            'noise_shape': noise_shape
        }
        return ret

    def encoder(self, Xs, Ys, net_shapes, reuse=False, name='encoder'):
        with tf.variable_scope(name, reuse=reuse):
            stack = Stacker(concat((flatten(Xs), Ys), axis=1))

            for shape in net_shapes:
                stack.linear_block(shape, relu)

            stack.linear_block(self.latent_code_size * 2, relu)

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
        self.noise = placeholder(tf.float32, self.noise_shape, name='noise')

        self.Xs_noised = tf.add(self.Xs, self.noise, name='Xs_noised')
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

    def train(self, Xs, Ys, epoch=100, save_interval=None, batch_size=None):
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

                Xs, Ys = dataset.next_batch(batch_size, batch_keys=['Xs', 'Ys'])
                noise = self.get_noises(Xs.shape)
                self.sess.run(self._train_ops, feed_dict={self._Xs: Xs, self._Ys: Ys, self._noise: noise})

            Xs, Ys = dataset.next_batch(batch_size, batch_keys=['Xs', 'Ys'], look_up=False)
            noise = self.get_noises(Xs.shape)
            loss = self.sess.run(self._metric_ops, feed_dict={self._Xs: Xs, self._Ys: Ys, self._noise: noise})
            self.log.info("e:{e} loss : {loss}".format(e=e, loss=np.mean(loss)))

            if save_interval is not None and e % save_interval == 0:
                self.save()

    def get_noises(self, shape=None):
        if shape is None:
            shape = self.Xs_shape
        return np.random.normal(-1 * self.noise_intensity, 1 * self.noise_intensity, size=shape)

    def code(self, Xs, Ys):
        noise = self.get_noises(Xs.shape)
        return self.get_tf_values(self._code_ops, {self._Xs: Xs, self.Ys: Ys, self._noise: noise})

    def recon(self, Xs, Ys):
        noise = self.get_noises(Xs.shape)
        return self.get_tf_values(self._recon_ops, {self._Xs: Xs, self.Ys: Ys, self._noise: noise})

    def metric(self, Xs, Ys):
        noise = self.get_noises(Xs.shape)
        return self.get_tf_values(self._metric_ops, {self._Xs: Xs, self.Ys: Ys, self._noise: noise})

    def generate(self, zs, Ys):
        return self.get_tf_values(self._recon_ops, {self._Ys: Ys, self._zs: zs})
