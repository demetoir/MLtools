import tensorflow as tf

from script.model.sklearn_like_model.NetModule.BaseNetModule import BaseNetModule
from script.model.sklearn_like_model.NetModule.DynamicVariable import DynamicVariable


class Momentum(BaseNetModule):
    def __init__(
            self,
            loss,
            train_var_list,
            learning_rate=0.001,
            momentum=0.9,
            reuse=False,
            name=None,
            verbose=0
    ):
        super().__init__(verbose=verbose, name=name, reuse=reuse)
        self.loss = loss
        self.train_var_list = train_var_list
        self._learning_rate = learning_rate
        self._momentum = momentum

        self._lr_module = DynamicVariable(self._learning_rate, name='learning_rate')

    def __str__(self):
        s = f"{self.__class__.__name__}"
        s += f"learning rate = {self.learning_rate}\n"
        s += f"momentum = {self._momentum}\n"

        return s

    @property
    def learning_rate(self):
        return self._lr_module.value

    def build(self):
        self.log.info(f'build {self.name}')
        with tf.variable_scope(self.name):
            self._lr_module.build()

            self._optimizer = tf.train.MomentumOptimizer(
                learning_rate=self._lr_module.variable,
                momentum=self._momentum
            )
            self._train_op = self.optimizer.minimize(
                loss=self.loss,
                var_list=self.train_var_list
            )

        return self

    def update_learning_rate(self, sess, lr):
        self._lr_module.update(sess, lr)

    @property
    def train_op(self):
        return self._train_op

    @property
    def optimizer(self):
        return self._optimizer