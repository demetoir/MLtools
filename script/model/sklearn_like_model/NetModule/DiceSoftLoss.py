from script.model.sklearn_like_model.NetModule.BaseLossModule import BaseLossModule
from script.util.tensor_ops import *


class DiceSoftLoss(BaseLossModule):
    def __init__(self, true, proba, smooth=1e-5, name=None, verbose=0, **kwargs):
        super().__init__(name, verbose, **kwargs)
        self.true = true
        self.proba = proba
        self.smooth = smooth

    @property
    def loss(self):
        return self.dice_loss

    def build(self):
        self.true_flatten = flatten(self.true)
        self.proba_flatten = flatten(self.proba)
        true = self.true_flatten
        proba = self.proba_flatten

        self.intersection = tf.reduce_sum(true * proba, axis=1, name='intersection')
        self.union = tf.reduce_sum((true * true) + (proba * proba), axis=1, name='union')
        # self.union = tf.reduce_sum(true + proba, axis=1)
        self.dice_coef = (2. * self.intersection + self.smooth) / (self.union + self.smooth)
        self.dice_coef = identity(self.dice_coef, 'dice_coef')
        self.dice_loss = 1 - self.dice_coef
        self.dice_loss = identity(self.dice_loss, 'dice_loss')
