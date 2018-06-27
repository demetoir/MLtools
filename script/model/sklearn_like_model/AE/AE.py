from tqdm import trange
from script.model.sklearn_like_model.Mixin import Xs_MixIn, zs_MixIn, noise_MixIn
from script.model.sklearn_like_model.BaseModel import BaseModel
from script.util.Stacker import Stacker
from script.util.tensor_ops import *
from script.util.summary_func import *
import numpy as np


def basicAE_Encoder(Xs, net_shapes, latent_code_size, reuse=False, name='encoder'):
    with tf.variable_scope(name, reuse=reuse):
        stack = Stacker(Xs)
        stack.flatten()
        for shape in net_shapes:
            stack.linear_block(shape, relu)

        stack.linear_block(latent_code_size, relu)

    return stack.last_layer


def basicAE_Decoder(zs, net_shapes, flatten_size, output_shape, reuse=False, name='decoder'):
    with tf.variable_scope(name, reuse=reuse):
        stack = Stacker(zs)
        for shape in net_shapes:
            stack.linear_block(shape, relu)

        stack.linear_block(flatten_size, sigmoid)
        stack.reshape(output_shape)

    return stack.last_layer


class basicAEPropertyMixIn(Xs_MixIn, zs_MixIn, noise_MixIn):
    def __init__(self):
        Xs_MixIn.__init__(self)
        zs_MixIn.__init__(self)
        noise_MixIn.__init__(self)

    @property
    def _train_ops(self):
        return [
            getattr(self, 'train_op'),
            getattr(self, 'op_inc_global_step')
        ]

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
    def _metric_ops(self):
        return getattr(self, 'loss')


class AE(BaseModel, basicAEPropertyMixIn):
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
        BaseModel.__init__(self, verbose)
        basicAEPropertyMixIn.__init__(self)

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

    def _build_input_shapes(self, shapes):
        ret = {}
        ret.update(self._build_Xs_input_shape(shapes))

        shapes['zs'] = [None, self.z_size]
        ret.update(self._build_zs_input_shape(shapes))

        shapes['noise'] = [None] + list(ret['X_shape'])
        ret.update(self._build_noise_input_shape(shapes))

        return ret

    def _build_main_graph(self):
        self.Xs = placeholder(tf.float32, self.Xs_shape, name='Xs')
        self.zs = placeholder(tf.float32, self.zs_shape, name='zs')
        self.noises = placeholder(tf.float32, self.noises_shape, name='noise')

        self.Xs_noised = tf.add(self.Xs, self.noises, name='Xs_noised')
        if self.with_noise:
            Xs = self.Xs_noised
        else:
            Xs = self.Xs

        self.latent_code = basicAE_Encoder(Xs, self.encoder_net_shapes, self.latent_code_size)
        self.Xs_recon = basicAE_Decoder(self.latent_code, self.decoder_net_shapes, self.X_flatten_size, self.Xs_shape)
        self.Xs_gen = basicAE_Decoder(self.zs, self.decoder_net_shapes, self.X_flatten_size, self.Xs_shape, reuse=True)

        head = get_scope()
        self.vars = collect_vars(join_scope(head, 'encoder'))
        self.vars += collect_vars(join_scope(head, 'decoder'))

    def _build_loss_function(self):
        self.loss = tf.squared_difference(self.Xs, self.Xs_recon, name='loss')
        self.loss_mean = tf.reduce_mean(self.loss, name='loss_mean')

    def _build_train_ops(self):
        self.train_op = tf.train.AdamOptimizer(self.learning_rate, self.beta1).minimize(loss=self.loss,
                                                                                        var_list=self.vars)

    def train(self, Xs, epoch=100, save_interval=None, batch_size=None):
        self._prepare_train(Xs=Xs)
        dataset = self.to_dummyDataset(Xs=Xs)

        if batch_size is None:
            batch_size = self.batch_size

        iter_num = 0
        iter_per_epoch = dataset.size // batch_size
        self.log.info("train epoch {}, iter/epoch {}".format(epoch, iter_per_epoch))
        for e in trange(epoch):
            dataset.shuffle()
            for i in range(iter_per_epoch):
                iter_num += 1

                Xs = dataset.next_batch(batch_size, batch_keys=['Xs'])
                noise = self.get_noises(Xs.shape)
                self.sess.run(self._train_ops, feed_dict={self._Xs: Xs, self._noises: noise})

            Xs = dataset.next_batch(batch_size, batch_keys=['Xs'], look_up=False)
            loss = self.metric(Xs)
            self.log.info("e:{e} loss : {loss}".format(e=e, loss=np.mean(loss)))

            if save_interval is not None and e % save_interval == 0:
                self.save()

    def code(self, Xs):
        noise = self.get_noises(Xs.shape)
        return self.get_tf_values(self._code_ops, {self._Xs: Xs, self._noises: noise})

    def recon(self, Xs):
        noise = self.get_noises(Xs.shape)
        return self.get_tf_values(self._recon_ops, {self._Xs: Xs, self._noises: noise})

    def metric(self, Xs):
        noise = self.get_noises(Xs.shape)
        return self.get_tf_values(self._metric_ops, {self._Xs: Xs, self._noises: noise})

    def generate(self, zs):
        return self.get_tf_values(self._recon_ops, {self._zs: zs})
