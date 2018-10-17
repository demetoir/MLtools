from script.util.MixIn import LoggerMixIn
from script.util.tensor_ops import *


class TFDynamicLearningRate(LoggerMixIn):

    def __init__(self, init_value, decay_method=None, decay=None, name='TFDynamicLearningRate', verbose=0):
        super().__init__(verbose)
        self.init_value = init_value
        self.decay_method = decay_method
        self.decay = decay
        self.name = name
        self._lr = init_value

        self.ph = None
        self.var = None
        self.is_build = False

    def __str__(self):
        s = f'{self.name}\n'
        s += f"learning rate = {self.learning_rate}\n"
        s += f"init value = {self.init_value}\n"
        s += f"decay_method = {self.decay_method}\n"
        s += f"decay = {self.decay}\n"
        s += f"is_build = {self.is_build}\n"
        return s

    @property
    def learning_rate(self):
        return self._lr

    @property
    def lr_var(self):
        if not self.is_build:
            self.build()

        return self.var

    @property
    def lr_ph(self):
        if not self.is_build:
            self.build()

        return self.ph

    @property
    def placeholder(self):
        return self.ph

    @property
    def variable(self):
        return self.var

    def build(self):
        self.log.info(f'build {self.name}')
        with tf.variable_scope(self.name):
            self.ph = placeholder(tf.float32, [], name=f'ph_{self.name}')
            self.var = tf.Variable(self.init_value, dtype=tf.float32, name=f'{self.name}')
            self.assign_op = tf.assign(self.var, self.ph)
            self.is_build = True

        return self

    def update(self, sess, x):
        sess.run(self.assign_op, feed_dict={self.ph: x})
        self._lr = x

    def lr_tensor(self, sess):
        return sess.run(self.lr_var)
