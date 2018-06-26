from tqdm import tqdm
from script.model.sklearn_like_model.BaseModel import BaseModel
from functools import reduce

from script.model.sklearn_like_model.DummyDataset import DummyDataset
from script.util.Stacker import Stacker
from script.util.tensor_ops import *
from script.util.summary_func import *
import numpy as np


class CVAE(BaseModel):

    @property
    def hyper_param_key(self):
        return [
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

    @property
    def _Xs(self):
        return self.Xs

    @property
    def _Ys(self):
        return self.Ys

    @property
    def _zs(self):
        return self.zs

    @property
    def _train_ops(self):
        return [self.train_op, self.op_inc_global_step]

    @property
    def _code_ops(self):
        return self.latent_code

    @property
    def _recon_ops(self):
        return self.Xs_recon

    @property
    def _generate_ops(self):
        return self.Xs_gen

    @property
    def _metric_ops(self):
        return self.loss

    def hyper_parameter(self):
        self.batch_size = 100
        self.learning_rate = 0.01
        self.beta1 = 0.5
        self.L1_norm_lambda = 0.001
        self.K_average_top_k_loss = 10
        self.latent_code_size = 32
        self.z_size = 32
        self.encoder_net_shapes = [512, 256, 128]
        self.decoder_net_shapes = [128, 256, 512]

    def build_input_shapes(self, input_shapes):
        self.X_batch_key = 'Xs'
        self.X_shape = input_shapes[self.X_batch_key]
        self.Xs_shape = [None] + self.X_shape
        self.X_flatten_size = reduce(lambda x, y: x * y, self.X_shape)

        self.z_shape = [self.z_size]
        self.zs_shape = [None, self.z_size]

        self.Y_batch_key = 'Ys'
        self.Y_shape = input_shapes[self.Y_batch_key]
        self.Ys_shape = [None] + self.Y_shape

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

        self.h = self.encoder(self.Xs, self.Ys, self.encoder_net_shapes)

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

    def random_z(self):
        pass

    def train(self, Xs, Ys, epoch=100, save_interval=None, batch_size=None):
        self.is_built()

        dataset = DummyDataset()
        dataset.add_data('Xs', Xs)
        dataset.add_data('Ys', Ys)

        if batch_size is None:
            batch_size = self.batch_size

        iter_num = 0
        iter_per_epoch = dataset.size // batch_size
        self.log.info("train epoch {}, iter/epoch {}".format(epoch, iter_per_epoch))

        for e in tqdm(range(epoch)):
            for i in range(iter_per_epoch):
                iter_num += 1

                Xs, Ys = dataset.next_batch(batch_size, batch_keys=['Xs', 'Ys'])
                self.sess.run(self._train_ops, feed_dict={self._Xs: Xs, self._Ys: Ys})

            Xs, Ys = dataset.next_batch(batch_size, batch_keys=['Xs', 'Ys'], look_up=False)
            loss = self.sess.run(self._metric_ops, feed_dict={self._Xs: Xs, self._Ys: Ys})
            self.log.info("e:{e} loss : {loss}".format(e=e, loss=np.mean(loss)))

            if save_interval is not None and e % save_interval == 0:
                self.save()

    def code(self, Xs, Ys):
        return self.sess.run(self._code_ops, feed_dict={self._Xs: Xs, self._Ys: Ys})

    def recon(self, Xs, Ys):
        return self.sess.run(self._recon_ops, feed_dict={self._Xs: Xs, self._Ys: Ys})

    def generate(self, zs, Ys):
        return self.sess.run(self._recon_ops, feed_dict={self._zs: zs, self._Ys: Ys})

    def metric(self, Xs, Ys):
        return self.sess.run(self._metric_ops, feed_dict={self._Xs: Xs, self._Ys: Ys})
