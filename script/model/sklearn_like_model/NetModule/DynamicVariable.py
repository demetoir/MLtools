from script.util.MixIn import LoggerMixIn
from script.util.tensor_ops import *


class DynamicVariable(LoggerMixIn):

    def __init__(self, value, name=None, verbose=0):
        super().__init__(verbose)
        self._value = value

        if name is None:
            name = self.__class__.__name__
        self.name = name

        self._placeholder = None
        self._variable = None
        self._is_build = False

    def __str__(self):
        s = f'{self.name}\n'
        s += f"value = {self._value}\n"
        s += f"placeholder = {self.placeholder}\n"
        s += f"variable = {self.variable}\n"
        s += f"is_build = {self._is_build}\n"
        return s

    @property
    def placeholder(self):
        return self._placeholder

    @property
    def variable(self):
        return self._variable

    @property
    def value(self):
        return self._value

    def build(self):
        self.log.info(f'build {self.name}')
        with tf.variable_scope(self.name):
            self._placeholder = placeholder(tf.float32, [], name=f'ph_{self.name}')
            self._variable = tf.Variable(self._value, dtype=tf.float32, name=f'{self.name}')
            self.assign_op = tf.assign(self._variable, self._placeholder)
            self._is_build = True

        return self

    def update(self, sess, x):
        sess.run(self.assign_op, feed_dict={self._placeholder: x})
        self._value = x

    def tensor_value(self, sess):
        return sess.run(self.variable)
