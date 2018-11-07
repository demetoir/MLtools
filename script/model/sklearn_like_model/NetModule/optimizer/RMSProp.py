import tensorflow as tf

from script.model.sklearn_like_model.NetModule.optimizer.optimizer import optimizer


class RMSProp(optimizer):
    def __init__(
            self,
            learning_rate=0.001,
            decay=0.9,
            momentum=0.0,
            epsilon=1e-8,
            use_locking=False,
            reuse=False,
            name=None,
            verbose=0
    ):
        super().__init__(learning_rate=learning_rate, verbose=verbose, name=name, reuse=reuse)

        self.decay = decay
        self.momentum = momentum
        self.epsilon = epsilon
        self.use_locking = use_locking

    def __str__(self):
        s = f"{self.__class__.__name__}"
        s += f"learning rate = {self.learning_rate}\n"
        s += f"decay = {self.decay}\n"
        s += f"momentum = {self.momentum}\n"
        s += f"epsilon = {self.epsilon}\n"
        s += f"use_lock = {self.use_locking}\n"
        return s

    def _build(self):
        self._optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self._lr_module.variable,
        )
        self._train_op = self.optimizer.minimize(
            loss=self.loss,
            var_list=self.train_var_list
        )
        return self._train_op
