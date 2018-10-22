import tensorflow as tf

from script.model.sklearn_like_model.NetModule.BaseNetModule import BaseNetModule
from script.model.sklearn_like_model.NetModule.DynamicVariable import DynamicVariable


class RMSProp(BaseNetModule):
    def __init__(
            self,
            loss,
            train_var_list,
            learning_rate=0.001,
            decay=0.9,
            momentum=0.0,
            epsilon=1e-8,
            use_locking=False,
            reuse=False,
            name=None,
            verbose=0
    ):
        super().__init__(verbose=verbose, name=name, reuse=reuse)
        self.loss = loss
        self.train_var_list = train_var_list
        self._learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.epsilon = epsilon
        self.use_locking = use_locking

        self._lr_module = DynamicVariable(self._learning_rate, name='learning_rate')

    def __str__(self):
        s = f"{self.__class__.__name__}"
        s += f"learning rate = {self.learning_rate}\n"
        s += f"decay = {self.decay}\n"
        s += f"momentum = {self.momentum}\n"
        s += f"epsilon = {self.epsilon}\n"
        s += f"use_lock = {self.use_locking}\n"
        return s

    def build(self):
        self.log.info(f'build {self.name}')
        with tf.variable_scope(self.name):
            self._lr_module.build()

            self._optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self._lr_module.variable,
            )
            self._train_op = self.optimizer.minimize(
                loss=self.loss,
                var_list=self.train_var_list
            )

        return self

    def update_learning_rate(self, sess, lr):
        self.lr_module.update(sess, lr)

    def reset_momentum(self, sess):
        if hasattr(self, 'init_momentum_op'):
            self.init_momentum_op = tf.initialize_variables(self.vars)

        sess.run(self.init_momentum_op)

    @property
    def train_op(self):
        return self._train_op

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def lr_module(self):
        return self._lr_module

    @property
    def learning_rate(self):
        return self.lr_module.value
