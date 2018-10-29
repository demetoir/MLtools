import numpy as np

from script.model.sklearn_like_model.BaseModel import BaseModel
from script.model.sklearn_like_model.NetModule.FusionNetStructure import FusionNetModule
from script.model.sklearn_like_model.NetModule.InceptionUNetModule import InceptionUNetModule
from script.model.sklearn_like_model.NetModule.UNetModule import UNetModule
from script.model.sklearn_like_model.NetModule.loss.BCELoss import BCELoss
from script.model.sklearn_like_model.NetModule.loss.DiceSoftLoss import DiceSoftLoss
from script.model.sklearn_like_model.NetModule.optimizer.NAG import NAG
from script.util.tensor_ops import *
from script.workbench.TGS_salt.lovazs_loss import lovasz_hinge


class SemanticSegmentation(BaseModel):
    net_structure_class_dict = {
        'UNet': UNetModule,
        'FusionNet': FusionNetModule,
        'InceptionUNet': InceptionUNetModule,
    }

    def __init__(
            self,
            verbose=10,
            learning_rate=0.01,
            beta1=0.9,
            batch_size=100,
            stage=4,
            net_type='UNet',
            loss_type='pixel_wise_softmax',
            n_classes=2,
            capacity=64,
            depth=1,
            dropout_rate=0.5,
            **kwargs
    ):
        BaseModel.__init__(self, verbose, **kwargs)

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.batch_size = batch_size
        self.stage = stage
        self.net_type = net_type
        self.loss_type = loss_type
        self.n_classes = n_classes
        self.capacity = capacity
        self.depth = depth
        self.dropout_rate = dropout_rate

    def _build_main_graph(self):
        self.Xs_ph = self.Xs
        self.Ys_ph = self.Ys

        net_class = self.net_structure_class_dict[self.net_type]
        self.net_module = net_class(
            self.Xs_ph,
            capacity=self.capacity, depth=self.depth, level=self.stage,
            n_classes=self.n_classes
        )
        self.net_module.build()
        self.vars = self.net_module.vars
        self._logit = self.net_module.logit
        self._proba = self.net_module.proba
        self._predict = tf.round(self._proba)

        self._predict_proba_ops = self._proba
        self._predict_ops = self._predict

    def _build_loss_ops(self):
        if self.loss_type == 'BCE+dice_soft':
            self.dice_soft_module = DiceSoftLoss(self.Ys_ph, self._proba)
            self.dice_soft_module.build()
            self.dice_soft = self.dice_soft_module.loss

            self.BCE_module = BCELoss(self.Ys_ph, self._logit)
            self.BCE_module.build()
            self.BCE = self.BCE_module.loss

            self.loss = self.dice_soft + self.BCE
            self.loss = self.BCE
            # self.loss = self.BCE

            lovasz = lovasz_hinge(self._logit, self.Ys_ph)

        else:
            raise NotImplementedError()

        def empty_mask_penalty(trues, predicts, batch_size, weight=0.1):
            penalty = []
            for i in range(batch_size):
                if np.sum(trues[i]) == 0:
                    penalty += [np.sum(predicts[i])]
                else:
                    penalty += [0]

            return np.mean(penalty) * weight

        # self.empty_penalty = empty_mask_penalty(self.Ys, self._predict, self.batch_size)
        # self.loss += self.empty_penalty

        def small_mask_penalty(trues, predicts, weight=0.1):
            predicts = tf.cast(flatten(predicts), tf.float32)
            trues = tf.cast(flatten(trues), tf.float32)

            inter = predicts * trues
            union = predicts + trues - predicts * trues
            loss = 1 - (inter / union)

            mask = tf.cast(tf.logical_and(trues < 0.05, trues > 0), tf.float32, name='loss mask')
            penalty = loss * mask
            return tf.reduce_mean(penalty * mask)

        # self.small_mask_penalty = small_mask_penalty(self.Ys, self._predict)
        # self.loss += self.small_mask_penalty
        return self.loss

    def _build_train_ops(self):
        self.optimizer = NAG(self.loss, self.vars, learning_rate=self.learning_rate)
        self.optimizer.build()
        self.train_ops = self.optimizer.train_op

        return self.train_ops

    def _build_metric_ops(self):
        return []
        pass

    def _build_predict_ops(self):
        return []
