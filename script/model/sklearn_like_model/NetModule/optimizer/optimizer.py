from script.model.sklearn_like_model.NetModule.BaseNetModule import BaseNetModule
from script.model.sklearn_like_model.NetModule.DynamicVariable import DynamicVariable
import tensorflow as tf


class optimizer(BaseNetModule):
    def __init__(
            self,
            learning_rate=0.001,
            reuse=False,
            name=None,
            verbose=0
    ):
        super().__init__(verbose=verbose, name=name, reuse=reuse)
        self._learning_rate = learning_rate
        self._optimizer = None
        self._train_op = None

        self._lr_module = DynamicVariable(self._learning_rate, name='learning_rate')

    @property
    def learning_rate(self):
        return self._lr_module.value

    @property
    def train_op(self):
        return self._train_op

    @property
    def optimizer(self):
        return self._optimizer

    def minimize(self, loss, var_list):
        self.loss = loss
        self.train_var_list = var_list

        return self

    def update_learning_rate(self, sess, lr):
        self._lr_module.update(sess, lr)

    def build(self):
        self.log.info(f'build {self.name}')
        with tf.variable_scope(self.name):
            self._lr_module.build()

            self._train_op = self._build()

        return self

    def _build(self):
        raise NotImplementedError
