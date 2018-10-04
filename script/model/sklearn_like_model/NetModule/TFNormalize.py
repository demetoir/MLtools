from script.util.MixIn import LoggerMixIn
from script.util.tensor_ops import *


class TFLnNormalize(LoggerMixIn):
    def __init__(self, var_list, rate=1.0, name='LnNormalize', verbose=0):
        super().__init__(verbose)
        self.var_list = var_list
        self.rate = rate
        self.name = name
        self._penalty = None
        self.is_build = False

    def normalize_func(self, var_list, rate):
        raise NotImplementedError

    def build(self):
        with tf.variable_scope(self.name):
            self.rate_ph = placeholder(tf.float32, [], name='rate_ph')
            self.rate_var = tf.Variable(self.rate, dtype=tf.float32, name='rate_var')
            self.update_op = tf.assign(self.rate_var, self.rate_ph)

            self._penalty = self.normalize_func(self.var_list, self.rate_var)
            self._penalty_mean = tf.reduce_mean(self._penalty, name='penalty_mean')
            self.is_build = True
            self.log.info(f'build {self.name}')

    def build_if_does_not(self):
        if not self.is_build:
            self.build()

    def update_rate(self, sess, rate):
        self.rate = rate

        self.build_if_does_not()
        sess.run(self.update_op, feed_dict={self.rate_ph: rate})

    @property
    def penalty(self):
        self.build_if_does_not()
        return self._penalty


class TFL1Normalize(TFLnNormalize):
    def __init__(self, var_list, rate=1.0, name='L1Normalize', verbose=0):
        super().__init__(var_list, rate, name, verbose)

    def normalize_func(self, var_list, rate):
        return L1_norm(var_list, lambda_=rate)


class TFL2Normalize(TFLnNormalize):
    def __init__(self, var_list, rate=1.0, name='L2Normalize', verbose=0):
        super().__init__(var_list, rate, name, verbose)

    def normalize_func(self, var_list, rate):
        return L2_norm(var_list, lambda_=rate)
