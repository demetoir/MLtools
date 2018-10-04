from script.model.sklearn_like_model.NetModule.DynamicDropoutRate import DynamicDropoutRate
from script.model.sklearn_like_model.NetModule.BaseNetModule import BaseNetModule
from script.util.Stacker import Stacker
from script.util.tensor_ops import *


class MLPNetModule(BaseNetModule):
    def __init__(self, x, n_classes, capacity=None, dropout_rate=0.5, depth=2, reuse=False, name=None, verbose=0):
        super().__init__(capacity, reuse, name, verbose)
        self.x = x
        self.n_classes = n_classes
        self.depth = depth
        self.dropout_rate = dropout_rate

        if capacity:
            self.capacity = capacity
        else:
            self.capacity = 1024

    def build(self):
        with tf.variable_scope(self.name, reuse=self.reuse):
            self.DDR = DynamicDropoutRate(self.dropout_rate).build()
            self.dropout_rate_tensor = self.DDR.tensor

            stack = Stacker(self.x)
            for _ in range(self.depth):
                stack.linear_block(self.capacity, relu)
                stack.dropout(self.dropout_rate_tensor)

            self.logit = stack.linear(self.n_classes)
            self.proba = stack.softmax()
        return self

    def set_train(self, sess):
        self.DDR.set_train(sess)

    def set_non_train(self, sess):
        self.DDR.set_non_train(sess)
