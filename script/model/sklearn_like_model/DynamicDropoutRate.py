from script.util.MixIn import LoggerMixIn
from script.util.tensor_ops import *


class DynamicDropoutRate(LoggerMixIn):
    def __init__(self, keep_prob=0.5, name='DropoutRate', verbose=0):
        super().__init__(verbose)

        self.keep_prob = keep_prob
        self.name = name

        self._ph = None
        self._tensor = None
        self.is_build = False

    @property
    def placeholder(self):
        if not self.is_build:
            self.build()

        return self._ph

    @property
    def tensor(self):
        if not self.is_build:
            self.build()

        return self._tensor

    def build(self):
        with tf.variable_scope(self.name):
            self._ph = placeholder(tf.float32, [], name=f'ph_{self.name}')
            self._tensor = tf.Variable(self.keep_prob, dtype=tf.float32, name=f'{self.name}')
            self.update_op = tf.assign(self._tensor, self._ph)
            self.is_build = True
            self.log.info(f'build {self.name}')

        return self

    def update(self, sess, x):
        return sess.run(self.update_op, feed_dict={self.placeholder: x})

    def set_train(self, sess):
        self.update(sess, self.keep_prob)

    def set_predict(self, sess):
        self.update(sess, 1)

    def eval(self, sess):
        return sess.run(self.tensor)
