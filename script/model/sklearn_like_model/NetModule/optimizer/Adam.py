import tensorflow as tf

from script.model.sklearn_like_model.NetModule.optimizer.optimizer import optimizer


class Adam(optimizer):

    def __init__(
            self,
            learning_rate=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            use_locking=False,
            reuse=False,
            name=None,
            verbose=0
    ):
        super().__init__(learning_rate=learning_rate, verbose=verbose, name=name, reuse=reuse)
        self._beta1 = beta1
        self._beta2 = beta2
        self.epsilon = epsilon
        self.use_locking = use_locking

    def __str__(self):
        s = f"{self.__class__.__name__}"
        s += f"learning rate = {self.learning_rate}\n"
        s += f"beta1 = {self._beta1}\n"
        s += f"beta2 = {self._beta2}\n"
        s += f"epsilon = {self.epsilon}\n"
        s += f"use_lock = {self.use_locking}\n"
        return s

    def _build(self):
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
        return self._train_op

    def reset_momentum(self, sess):
        if hasattr(self, 'init_momentum_op'):
            self.init_momentum_op = tf.initialize_variables(self.vars)

        sess.run(self.init_momentum_op)
