import tensorflow as tf

from script.model.sklearn_like_model.NetModule.optimizer.optimizer import optimizer


class Adagrad(optimizer):
    def __init__(
            self,

            learning_rate,
            initial_accumulator_value=0.1,
            reuse=False,
            name=None,
            verbose=0
    ):
        super().__init__(learning_rate=learning_rate, verbose=verbose, name=name, reuse=reuse)
        self.initial_accumulator_value = initial_accumulator_value

    def __str__(self):
        s = f"{self.__class__.__name__}"
        s += f"learning rate = {self.learning_rate}\n"
        s += f"initial_accumulator_value = {self.initial_accumulator_value}\n"
        return s

    def _build(self):
        self._optimizer = tf.train.AdagradOptimizer(
            learning_rate=self._lr_module.variable,
            initial_accumulator_value=self.initial_accumulator_value,
        )
        self._train_op = self.optimizer.minimize(
            loss=self.loss,
            var_list=self.train_var_list
        )
        return self._train_op
