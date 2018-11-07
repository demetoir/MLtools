import tensorflow as tf

from script.model.sklearn_like_model.NetModule.optimizer.optimizer import optimizer


class AdagradDA(optimizer):
    def __init__(
            self,
            learning_rate,
            global_step,
            initial_gradient_squared_accumulator_value=0.1,
            l1_regularization_strength=0.0,
            l2_regularization_strength=0.0,
            reuse=False,
            name=None,
            verbose=0
    ):
        super().__init__(learning_rate=learning_rate, verbose=verbose, name=name, reuse=reuse)
        self.global_step = global_step
        self.initial_gradient_squared_accumulator_value = initial_gradient_squared_accumulator_value
        self.l1_regularization_strength = l1_regularization_strength
        self.l2_regularization_strength = l2_regularization_strength

    def __str__(self):
        s = f"{self.__class__.__name__}"
        s += f"learning rate = {self.learning_rate}\n"
        s += f"global_step = {self.global_step}\n"
        s += f"initial_gradient_squared_accumulator_value = {self.initial_gradient_squared_accumulator_value}\n"
        s += f"l1_regularization_strength = {self.l1_regularization_strength}\n"
        s += f"l2_regularization_strength = {self.l2_regularization_strength}\n"

        return s

    def _build(self):
        self._optimizer = tf.train.AdagradDAOptimizer(
            learning_rate=self._lr_module.variable,
            global_step=self.global_step,
            initial_gradient_squared_accumulator_value=self.initial_gradient_squared_accumulator_value,
            l1_regularization_strength=self.l1_regularization_strength,
            l2_regularization_strength=self.l2_regularization_strength,
        )
        self._train_op = self.optimizer.minimize(
            loss=self.loss,
            var_list=self.train_var_list
        )
        return self._train_op
