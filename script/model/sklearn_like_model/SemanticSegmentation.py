import numpy as np
from script.model.sklearn_like_model.BaseModel import BaseModel
from script.model.sklearn_like_model.NetModule.loss.BCELoss import BCELoss
from script.model.sklearn_like_model.NetModule.loss.DiceSoftLoss import DiceSoftLoss
from script.model.sklearn_like_model.NetModule.FusionNetStructure import FusionNetModule
from script.model.sklearn_like_model.NetModule.InceptionUNetModule import InceptionUNetModule
from script.model.sklearn_like_model.NetModule.PlaceHolderModule import PlaceHolderModule
from script.model.sklearn_like_model.NetModule.TFDynamicLearningRate import TFDynamicLearningRate
from script.model.sklearn_like_model.NetModule.UNetModule import UNetModule
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

    def update_learning_rate(self, lr):
        self.learning_rate = lr

        if self.sess is not None:
            self.drl.update(self.sess, self.learning_rate)

    def update_dropout_rate(self, rate):
        self.net_module.update_dropout_rate(self.sess, rate)
        self.dropout_rate = rate

    def _build_input_shapes(self, shapes):
        self.xs_ph_module = PlaceHolderModule(shapes['x'], name='x').build()
        self.ys_ph_module = PlaceHolderModule(shapes['y'], name='y').build()

        ret = {}
        ret.update(self.xs_ph_module.shape_dict)
        ret.update(self.ys_ph_module.shape_dict)
        return ret

    def _build_main_graph(self):
        self.Xs_ph = self.xs_ph_module.placeholder
        self.Ys_ph = self.ys_ph_module.placeholder

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

    def _train_iter(self, dataset, batch_size):
        self.net_module.set_train(self.sess)

        Xs, Ys = dataset.next_batch(batch_size, balanced_class=False)
        self.sess.run(self.train_ops, feed_dict={self.Xs_ph: Xs, self.Ys_ph: Ys})

        self.net_module.set_non_train(self.sess)

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

    def _build_train_ops(self):
        self.drl = TFDynamicLearningRate(self.learning_rate)
        self.drl.build()

        self.train_ops = tf.train.AdamOptimizer(
            self.drl.learning_rate
        ).minimize(
            self.loss, var_list=self.vars
        )

    @property
    def predict_ops(self):
        return self._predict_ops

    @property
    def predict_proba_ops(self):
        return self._predict_proba_ops

    @property
    def score_ops(self):
        return self.metric_ops

    @property
    def metric_ops(self):
        return self.loss

    def init_adam_momentum(self):
        self.sess.run(tf.variables_initializer(self.train_ops_var_list))

    def _metric(self, x=None, y=None):
        return self.batch_execute(self.loss, {self.Xs_ph: x, self.Ys_ph: y})

    def predict_proba(self, x):
        return self.batch_execute(self._predict_proba_ops, {self.Xs_ph: x})

    def predict(self, x):
        return self.batch_execute(self._predict_ops, {self.Xs_ph: x})

    def score(self, x, y):
        return np.mean(self.batch_execute(self.score_ops, {self.Xs_ph: x, self.Ys_ph: y}))
