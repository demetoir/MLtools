import tensorflow as tf
from script.util.tensor_ops import identity


class GAN_loss_builder_MixIn:

    def __init__(self):
        cls = self.__class__
        self._loss_builder_funcs = {
            'WGAN': cls.WGAN_loss,
            'GAN': cls.GAN_loss,
            'LSGAN': cls.LSGAN_loss,
            'L1_GAN': cls.L1_GAN_loss
        }

    @staticmethod
    def WGAN_loss(D_real, D_gen):
        D_real_loss = identity(-tf.reduce_mean(D_real, axis=1), 'D_real_loss')
        D_gen_loss = identity(tf.reduce_mean(D_gen, axis=1), 'D_gen_loss')
        D_loss = identity(D_real_loss + D_gen_loss, 'D_loss')
        G_loss = identity(-D_gen_loss, 'G_loss')

        return D_real_loss, D_gen_loss, D_loss, G_loss

    @staticmethod
    def GAN_loss(D_real, D_gen):
        D_real_loss = tf.reduce_mean(D_real, axis=1, name='D_real_loss')
        D_gen_loss = tf.reduce_mean(D_gen, axis=1, name='D_gen_loss')
        D_loss = identity(-tf.log(D_real) - tf.log(1. - D_gen), name='D_loss')
        G_loss = identity(-tf.log(D_gen), name='G_loss')

        return D_real_loss, D_gen_loss, D_loss, G_loss

    @staticmethod
    def LSGAN_loss(D_real, D_gen):
        D_real_loss = tf.reduce_mean(D_real, axis=1, name='D_real_loss')
        D_gen_loss = tf.reduce_mean(D_gen, axis=1, name='D_gen_loss')

        square_sum = tf.square(tf.subtract(D_real, 1)) + tf.square(D_gen)
        D_loss = identity(tf.multiply(0.5, square_sum), 'D_loss')
        G_loss = identity(tf.multiply(0.5, tf.square(tf.subtract(D_gen, 1))), 'G_loss')

        return D_real_loss, D_gen_loss, D_loss, G_loss

    @staticmethod
    def L1_GAN_loss(D_real, D_gen):
        D_real_loss = tf.reduce_mean(tf.abs(D_real - 1.), axis=1, name='D_real_loss')
        D_gen_loss = tf.reduce_mean(tf.abs(D_gen), axis=1, name='D_gen_loss')

        D_loss = identity(D_real_loss + D_gen_loss, 'D_loss')
        G_loss = identity(tf.abs(D_gen - 1.), 'G_loss')

        return D_real_loss, D_gen_loss, D_loss, G_loss

    def _build_GAN_loss(self, D_real, D_gen, loss_type='GAN'):
        D_real_loss, D_gen_loss, D_loss, G_loss = self._loss_builder_funcs[loss_type](D_real, D_gen)

        return D_real_loss, D_gen_loss, D_loss, G_loss
