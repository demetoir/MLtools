import tensorflow as tf

from script.model.sklearn_like_model.NetModule.BaseNetModule import BaseNetModule
from script.model.sklearn_like_model.NetModule.DynamicVariable import DynamicVariable


class Adam(BaseNetModule):
    def __init__(
            self,
            loss,
            train_var_list,
            learning_rate=0.001,
            beta1=0.9,
            beta2=0.999,
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
        self._beta1 = beta1
        self._beta2 = beta2
        self.epsilon = epsilon
        self.use_locking = use_locking

        self._lr_module = DynamicVariable(self._learning_rate, name='learning_rate')

    def __str__(self):
        s = f"{self.__class__.__name__}"
        s += f"learning rate = {self.learning_rate}\n"
        s += f"beta1 = {self._beta1}\n"
        s += f"beta2 = {self._beta2}\n"
        s += f"epsilon = {self.epsilon}\n"
        s += f"use_lock = {self.use_locking}\n"
        return s

    @property
    def learning_rate(self):
        return self._lr_module.value

    def build(self):
        self.log.info(f'build {self.name}')
        with tf.variable_scope(self.name):
            self._lr_module.build()

            self._optimizer = tf.train.AdamOptimizer(
                learning_rate=self._lr_module.variable,
                beta1=self._beta1,
                beta2=self._beta2,
                epsilon=self.epsilon,
                use_locking=self.use_locking
            )
            self._train_op = self.optimizer.minimize(
                loss=self.loss,
                var_list=self.train_var_list
            )

        return self

    def update_learning_rate(self, sess, lr):
        self._lr_module.update(sess, lr)

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