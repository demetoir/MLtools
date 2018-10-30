import tensorflow as tf

from script.model.sklearn_like_model.NetModule.optimizer.optimizer import optimizer


class NAG(optimizer):
    def __init__(
            self,
            learning_rate=0.001,
            momentum=0.9,
            reuse=False,
            name=None,
            verbose=0
    ):
        super().__init__(learning_rate=learning_rate, verbose=verbose, name=name, reuse=reuse)
        self._momentum = momentum

    def __str__(self):
        s = f"{self.__class__.__name__}"
        s += f"learning rate = {self.learning_rate}\n"
        s += f"momentum = {self._momentum}\n"
        return s

    def _build(self):
        self._optimizer = tf.train.MomentumOptimizer(
            learning_rate=self._lr_module.variable,
            momentum=self._momentum,
            use_nesterov=True
        )
        self._train_op = self.optimizer.minimize(
            loss=self.loss,
            var_list=self.train_var_list
        )
        return self._train_op
