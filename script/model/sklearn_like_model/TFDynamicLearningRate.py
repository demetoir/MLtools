from script.util.MixIn import LoggerMixIn
from script.util.Stacker import Stacker
from script.util.tensor_ops import *


class TFDynamicLearningRate(LoggerMixIn):

    def __init__(self, init_value, decay_method=None, decay=None, name='TFDynamicLearningRate', verbose=0):
        super().__init__(verbose)
        self.init_value = init_value
        self.decay_method = decay_method
        self.decay = decay
        self.name = name

        self._lr_ph = None
        self._lr_var = None
        self.is_build = False

    @property
    def learning_rate(self):
        if not self.is_build:
            self.build()

        return self._lr_var

    @property
    def lr_var(self):
        if not self.is_build:
            self.build()

        return self._lr_var

    @property
    def lr_ph(self):
        if not self.is_build:
            self.build()

        return self._lr_ph

    def build(self):
        with tf.variable_scope(self.name):
            self._lr_ph = placeholder(tf.float32, [], name=f'ph_{self.name}')
            self._lr_var = tf.Variable(self.init_value, dtype=tf.float32, name=f'{self.name}')
            self.assign_op = tf.assign(self._lr_var, self._lr_ph)
            self.is_build = True
            self.log.info(f'build {self.name}')

    def update(self, sess, x):
        sess.run(self.assign_op, feed_dict={self._lr_ph: x})

    def lr_tensor(self, sess):
        return sess.run(self.lr_var)


def test_TFDynamicLearningRate():
    import numpy as np
    x = np.random.normal(size=[100, 10])
    y = np.random.normal(size=[100, 1])
    x_ph = placeholder(tf.float32, [-1, 10], name='ph_x')

    stack = Stacker(x_ph)
    stack.linear_block(100, relu)
    stack.linear_block(100, relu)
    logit = stack.linear(1)
    proba = stack.softmax()

    loss = (proba - y) ** 2
    dlr = TFDynamicLearningRate(0.01)
    dlr.build()

    lr_var = dlr.learning_rate
    var_list = None
    train_op = tf.train.AdamOptimizer(learning_rate=lr_var, beta1=0.9).minimize(loss, var_list=var_list)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(f'lr = {dlr.lr_tensor(sess)}')

        dlr.update(sess, 0.1)
        print(f'lr = {dlr.lr_tensor(sess)}')
        sess.run(train_op, feed_dict={x_ph: x})
        print(f'loss = {np.mean(sess.run(loss, feed_dict={x_ph: x}))}')

        dlr.update(sess, 0.05)
        print(f'lr = {dlr.lr_tensor(sess)}')
        sess.run(train_op, feed_dict={x_ph: x})
        print(f'loss = {np.mean(sess.run(loss, feed_dict={x_ph: x}))}')

        dlr.update(sess, 0.02)
        print(f'lr = {dlr.lr_tensor(sess)}')
        sess.run(train_op, feed_dict={x_ph: x})
        print(f'loss = {np.mean(sess.run(loss, feed_dict={x_ph: x}))}')
