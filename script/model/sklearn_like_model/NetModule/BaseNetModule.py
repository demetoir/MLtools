from tensorflow.contrib import slim
from script.util.MixIn import LoggerMixIn
from script.util.tensor_ops import collect_vars, join_scope, get_scope
import tensorflow as tf


def collect_global_vars(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)


def collect_trainable_vars(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)


class BaseNetModule(LoggerMixIn):
    def __init__(self, capacity=None, reuse=False, name=None, verbose=0):
        super().__init__(verbose)
        self.capacity = capacity
        self.reuse = reuse
        if name:
            self.name = name
        else:
            self.name = self.__class__.__name__
        self._vars = None
        self.scope = join_scope(get_scope(), self.name)

    @property
    def vars(self):
        if self._vars is None:
            if len(get_scope()) == 0:
                self._vars = collect_vars(self.name)
            else:
                self._vars = collect_vars(join_scope(get_scope(), self.name))
        return self._vars

    @property
    def var_list(self):
        return collect_global_vars(self.scope)

    @property
    def trainable_var_list(self):
        return collect_trainable_vars(self.scope)

    def build(self):
        raise NotImplementedError

    def show_vars_summary(self):
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)
