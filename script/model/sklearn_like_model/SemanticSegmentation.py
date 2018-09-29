from script.model.sklearn_like_model.BaseModel import BaseModel
from script.model.sklearn_like_model.Mixin import Xs_MixIn, Ys_MixIn, supervised_trainMethodMixIn, predictMethodMixIn, \
    predict_probaMethodMixIn, scoreMethodMixIn, supervised_metricMethodMixIn
from script.model.sklearn_like_model.NetModule.InceptionUNetModule import InceptionUNetModule
from script.model.sklearn_like_model.TFDynamicLearningRate import TFDynamicLearningRate
from script.model.sklearn_like_model.NetModule.FusionNetStructure import FusionNetModule
from script.model.sklearn_like_model.NetModule.UNetNetModule import UNetNetModule
from script.util.MixIn import LoggerMixIn
from script.util.tensor_ops import *
import numpy as np


class SemanticSegmentationLossModule(LoggerMixIn):
    def __init__(
            self,
            loss_type,
            labels=None,
            probas=None,
            logits=None,
            n_classes=2,
            name='SegmentationLoss',
            verbose=0
    ):
        super().__init__(verbose=verbose)
        self.loss_type = loss_type
        self.labels = labels
        self.probas = probas
        self.logits = logits
        self.n_classes = n_classes
        self.name = name

        cls = self.__class__

        self.loss_builder_func = {
            'iou': cls.build_iou,
            'dice_soft': cls.build_dice_soft,
            'pixel_wise_softmax': cls.build_pixel_wise_softmax
        }

        self._loss = None

    @property
    def loss(self):
        return self._loss

    def build(self):
        with tf.variable_scope(self.name + '_' + self.loss_type):
            if self.loss_type == 'iou':
                loss = self.build_iou()
            elif self.loss_type == 'dice_soft':
                loss = self.build_dice_soft()
            elif self.loss_type == 'pixel_wise_softmax':
                loss = self.build_pixel_wise_softmax()
            else:
                raise ValueError(f'{self.loss_type} is can not build')

            self._loss = loss
            self.log.info(f'build {self.name}_{self.loss_type}')

        return self.loss

    def build_iou(self):
        labels = self.labels
        probas = self.probas

        # only binary mask
        probas = probas[:, :, :, 1]

        # https://angusg.com/writing/2016/12/28/optimizing-iou-semantic-segmentation.html
        probas = tf.cast(tf.reshape(probas, [-1]), tf.float32)
        labels = tf.cast(tf.reshape(labels, [-1]), tf.float32)

        inter = tf.reduce_sum(probas * labels)
        union = tf.reduce_sum(probas + labels - probas * labels)
        loss = 1 - (inter / union)

        return loss

    def build_dice_soft(self):
        labels = self.labels
        probas = self.probas
        # only binary mask
        probas = probas[:, :, :, 1]

        probas = tf.cast(tf.reshape(probas, [-1]), tf.float32)
        labels = tf.cast(tf.reshape(labels, [-1]), tf.float32)

        inter = tf.reduce_sum(probas * labels)
        union = tf.reduce_sum(probas + labels)
        loss = 1 - (2 * inter / union)
        return loss

    def build_pixel_wise_softmax(self):
        labels = self.labels
        logits = self.logits
        n_classes = self.n_classes

        logits = tf.reshape(logits, (-1, n_classes))
        labels = tf.cast(tf.reshape(labels, [-1]), tf.int32)
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)


class SemanticSegmentation(
    BaseModel,
    Xs_MixIn,
    Ys_MixIn,
    supervised_trainMethodMixIn,
    predictMethodMixIn,
    predict_probaMethodMixIn,
    scoreMethodMixIn,
    supervised_metricMethodMixIn,
):
    net_structure_class_dict = {
        'UNet': UNetNetModule,
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
        Xs_MixIn.__init__(self)
        Ys_MixIn.__init__(self)
        supervised_trainMethodMixIn.__init__(self, None)
        predictMethodMixIn.__init__(self)
        predict_probaMethodMixIn.__init__(self)
        scoreMethodMixIn.__init__(self)
        supervised_metricMethodMixIn.__init__(self)

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

    def _build_input_shapes(self, shapes):
        ret = {}
        ret.update(self._build_Xs_input_shape(shapes))
        ret.update(self._build_Ys_input_shape(shapes))
        return ret

    def _build_main_graph(self):
        self.Xs = placeholder(tf.float32, self.Xs_shape, name='Xs')
        self.Ys = placeholder(tf.float32, self.Ys_shape, name='Ys')

        net_class = self.net_structure_class_dict[self.net_type]
        self.net_module = net_class(
            self.Xs,
            capacity=self.capacity, depth=self.depth, level=self.stage,
            n_classes=self.n_classes
        )
        self.net_module.build()
        self.vars = self.net_module.vars
        self._logit = self.net_module.logit
        self._proba = self.net_module.proba
        self._predict = reshape(tf.argmax(self._proba, 3, name="predicted"), self.Ys_shape,
                                name='predict')

        self._predict_proba_ops = self._proba
        self._predict_ops = self._predict

    def _build_loss_function(self):
        if self.loss_type == 'BCE+dice_soft':
            self.dice_soft_loss_module = SemanticSegmentationLossModule(
                loss_type='dice_soft', labels=self.Ys, logits=self._logit,
                probas=self._proba)
            self.dice_soft_loss_module.build()
            self.dice_soft = self.dice_soft_loss_module.loss

            self.pixel_wise_softmax_loss_module = SemanticSegmentationLossModule(
                loss_type='pixel_wise_softmax', labels=self.Ys,
                logits=self._logit,
                probas=self._proba)
            self.pixel_wise_softmax_loss_module.build()
            self.pixel_wise_softmax = self.pixel_wise_softmax_loss_module.loss

            self.loss = self.dice_soft + self.pixel_wise_softmax
        else:
            self.loss_module = SemanticSegmentationLossModule(loss_type=self.loss, labels=self.Ys, logits=self._logit,
                                                              probas=self._proba)
            self.loss_module.build()
            self.loss = self.loss_module.loss

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

        self._train_ops = tf.train.AdamOptimizer(
            self.drl.learning_rate, beta1=self.beta1
        ).minimize(
            self.loss, var_list=self.vars
        )

    @property
    def train_ops(self):
        return self._train_ops

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

    def _train_iter(self, dataset, batch_size):
        self.net_module.set_train(self.sess)

        Xs, Ys = dataset.next_batch(batch_size, balanced_class=False)
        self.sess.run(self.train_ops, feed_dict={self._Xs: Xs, self._Ys: Ys})

        self.net_module.set_non_train(self.sess)
