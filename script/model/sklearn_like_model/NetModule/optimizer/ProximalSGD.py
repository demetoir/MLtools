import tensorflow as tf

from script.model.sklearn_like_model.NetModule.optimizer.optimizer import optimizer


class ProximalSGD(optimizer):
    def __init__(
            self,
            learning_rate=0.001,
            l1_regularization_strength=0.0,
            l2_regularization_strength=0.0,
            reuse=False,
            name=None,
            verbose=0
    ):
        super().__init__(learning_rate=learning_rate, verbose=verbose, name=name, reuse=reuse)

        self._l1_regularization_strength = l1_regularization_strength
        self._l2_regularization_strength = l2_regularization_strength

    def __str__(self):
        s = f"{self.__class__.__name__}"
        s += f"learning rate = {self.learning_rate}\n"
        s += f"l1_regularization_strength = {self._l1_regularization_strength}\n"
        s += f"l2_regularization_strength= {self._l2_regularization_strength}\n"

        return s

    def _build(self):
        self._optimizer = tf.train.ProximalGradientDescentOptimizer(
            learning_rate=self._lr_module.variable,
            l1_regularization_strength=self._l1_regularization_strength,
            l2_regularization_strength=self._l2_regularization_strength,
        )
        self._train_op = self.optimizer.minimize(
            loss=self.loss,
            var_list=self.train_var_list
        )
        return self._train_op
