from script.model.sklearn_like_model.NetModule.BaseLossModule import BaseLossModule
from script.util.tensor_ops import *


class BCELoss(BaseLossModule):
    def __init__(self, true, logit, name=None, verbose=0, **kwargs):
        super().__init__(name, verbose, **kwargs)
        self.true = true
        self.logit = logit

    def _build(self):
        self.true_flatten = flatten(self.true)
        self.logit_flatten = flatten(self.logit)
        true = self.true_flatten
        logit = self.logit_flatten
        self.bce = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=true, logits=logit),
            axis=1)

    @property
    def loss(self):
        return self.bce
