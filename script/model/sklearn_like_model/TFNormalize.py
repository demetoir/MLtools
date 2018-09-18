from script.util.MixIn import LoggerMixIn
from script.util.Stacker import Stacker
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


def test_TFL1Normalize():
    import numpy as np
    x = np.random.normal(size=[100, 10])
    y = np.random.normal(size=[100, 1])
    x_ph = placeholder(tf.float32, [-1, 10], name='ph_x')

    with tf.variable_scope('net'):
        stack = Stacker(x_ph)
        stack.linear_block(100, relu)
        stack.linear_block(100, relu)
        logit = stack.linear(1)
        proba = stack.softmax()

        loss = (proba - y) ** 2

    var_list = collect_vars('net')
    l1_norm = TFL1Normalize(var_list)
    l1_norm.build()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        val = sess.run(l1_norm.penalty, feed_dict={x_ph: x})
        rate = sess.run(l1_norm.rate_var)
        print(f'l1_norm = {val}, rate = {rate}')

        l1_norm.update_rate(sess, 0.5)
        val = sess.run(l1_norm.penalty, feed_dict={x_ph: x})
        rate = sess.run(l1_norm.rate_var)
        print(f'l1_norm = {val}, rate = {rate}')

        l1_norm.update_rate(sess, 0.1)
        val = sess.run(l1_norm.penalty, feed_dict={x_ph: x})
        rate = sess.run(l1_norm.rate_var)
        print(f'l1_norm = {val}, rate = {rate}')


def test_TFL2Normalize():
    import numpy as np
    x = np.random.normal(size=[100, 10])
    y = np.random.normal(size=[100, 1])
    x_ph = placeholder(tf.float32, [-1, 10], name='ph_x')

    with tf.variable_scope('net'):
        stack = Stacker(x_ph)
        stack.linear_block(100, relu)
        stack.linear_block(100, relu)
        logit = stack.linear(1)
        proba = stack.softmax()

        loss = (proba - y) ** 2

    var_list = collect_vars('net')
    l2_norm = TFL2Normalize(var_list)
    l2_norm.build()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        val = sess.run(l2_norm.penalty, feed_dict={x_ph: x})
        rate = sess.run(l2_norm.rate_var)
        print(f'l2_norm = {val}, rate = {rate}')

        l2_norm.update_rate(sess, 0.5)
        val = sess.run(l2_norm.penalty, feed_dict={x_ph: x})
        rate = sess.run(l2_norm.rate_var)
        print(f'l2_norm = {val}, rate = {rate}')

        l2_norm.update_rate(sess, 0.1)
        val = sess.run(l2_norm.penalty, feed_dict={x_ph: x})
        rate = sess.run(l2_norm.rate_var)
        print(f'l2_norm = {val}, rate = {rate}')
