from script.util.MixIn import LoggerMixIn
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


