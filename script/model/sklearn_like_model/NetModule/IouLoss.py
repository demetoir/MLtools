from script.model.sklearn_like_model.NetModule.BaseLossModule import BaseLossModule
from script.util.tensor_ops import *


class IouLoss(BaseLossModule):
    def __init__(self, true, proba, smooth=1e-5, name=None, verbose=0, **kwargs):
        super().__init__(name, verbose, **kwargs)
        self.true = true
        self.proba = proba
        self.smooth = smooth

    def build(self):
        self.true_flatten = flatten(self.true)
        self.proba_flatten = flatten(self.proba)
        true = self.true_flatten
        proba = self.proba_flatten

        self.intersection = tf.reduce_sum(true * proba, axis=1)
        self.union = tf.reduce_sum((true * true) + (proba * proba), axis=1)
        self.iou_coef = (self.intersection + self.smooth) / (self.union - self.intersection + self.smooth)
        self.iou_loss = 1 - self.iou_coef

    @property
    def loss(self):
        return self.iou_loss
