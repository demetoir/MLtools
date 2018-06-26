from script.model.sklearn_like_model.AE.BaseAutoEncoder import BaseAutoEncoder
from script.util.Stacker import Stacker
from script.util.tensor_ops import *
from script.util.summary_func import *
from functools import reduce


class basicAEEncoderMixIn:
    @staticmethod
    def encoder(Xs, net_shapes, latent_code_size, reuse=False, name='encoder'):
        with tf.variable_scope(name, reuse=reuse):
            stack = Stacker(Xs)
            stack.flatten()
            for shape in net_shapes:
                stack.linear_block(shape, relu)

            stack.linear_block(latent_code_size, relu)

        return stack.last_layer


class basicAEDecoderMixIn:
    @staticmethod
    def decoder(zs, net_shapes, flatten_size, output_shape, reuse=False, name='decoder'):
        with tf.variable_scope(name, reuse=reuse):
            stack = Stacker(zs)
            for shape in net_shapes:
                stack.linear_block(shape, relu)

            stack.linear_block(flatten_size, sigmoid)
            stack.reshape(output_shape)

        return stack.last_layer


class AutoEncoder(BaseAutoEncoder, basicAEEncoderMixIn, basicAEDecoderMixIn):
    _input_shape_keys = [
        'X_shape',
        'Xs_shape',
        'X_flatten_size',
        'z_shape',
        'zs_shape'
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
        'decoder_net_shapes'
    ]

    def __init__(self, verbose=10):
        super(AutoEncoder, self).__init__(verbose)

        self.batch_size = 100
        self.learning_rate = 0.01
        self.beta1 = 0.5
        self.L1_norm_lambda = 0.001
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

        X_flatten_size = reduce(lambda x, y: x * y, X_shape)
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

        self.latent_code = self.encoder(self.Xs, self.encoder_net_shapes, self.latent_code_size)
        self.Xs_recon = self.decoder(self.latent_code, self.decoder_net_shapes, self.X_flatten_size, self.Xs_shape)
        self.Xs_gen = self.decoder(self.zs, self.decoder_net_shapes, self.X_flatten_size, self.Xs_shape, reuse=True)

        head = get_scope()
        self.vars = collect_vars(join_scope(head, 'encoder'))
        self.vars += collect_vars(join_scope(head, 'decoder'))

    def _build_loss_function(self):
        self.loss = tf.squared_difference(self.Xs, self.Xs_recon, name='loss')
        self.loss_mean = tf.reduce_mean(self.loss, name='loss_mean')

    def _build_train_ops(self):
        self.train_op = tf.train.AdamOptimizer(self.learning_rate, self.beta1).minimize(loss=self.loss,
                                                                                        var_list=self.vars)

    def random_z(self):
        pass
