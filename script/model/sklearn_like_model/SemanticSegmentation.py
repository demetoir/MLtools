from script.model.sklearn_like_model.BaseModel import BaseModel
from script.model.sklearn_like_model.Mixin import Xs_MixIn, Ys_MixIn, supervised_trainMethodMixIn, predictMethodMixIn, \
    predict_probaMethodMixIn, scoreMethodMixIn, supervised_metricMethodMixIn
from script.model.sklearn_like_model.TFDynamicLearningRate import TFDynamicLearningRate
from script.model.sklearn_like_model.net_structure.FusionNetStructure import FusionNetStructure
from script.model.sklearn_like_model.net_structure.UNetStructure import UNetStructure
from script.util.tensor_ops import *


class segmentation_loss_mixIn:

    def __init__(self):
        cls = self.__class__

        self.loss_builder_func = {
            'iou': cls._iou,
            'dice_soft': cls._dice_soft,
            'pixel_wise_softmax': cls._pixel_wise_softmax
        }

    def _build_loss(self, loss_type, **kwargs):
        with tf.variable_scope(loss_type + '_loss'):
            return self.loss_builder_func[loss_type](**kwargs)

    @staticmethod
    def _iou(labels=None, probas=None, **kwargs):
        # only binary mask
        probas = probas[:, :, :, 1]

        # https://angusg.com/writing/2016/12/28/optimizing-iou-semantic-segmentation.html
        probas = tf.cast(tf.reshape(probas, [-1]), tf.float32)
        labels = tf.cast(tf.reshape(labels, [-1]), tf.float32)

        inter = tf.reduce_sum(probas * labels)
        union = tf.reduce_sum(probas + labels - probas * labels)
        loss = 1 - (inter / union)
        return loss

    @staticmethod
    def _dice_soft(labels, probas, **kwargs):
        # only binary mask
        probas = probas[:, :, :, 1]

        probas = tf.cast(tf.reshape(probas, [-1]), tf.float32)
        labels = tf.cast(tf.reshape(labels, [-1]), tf.float32)

        inter = tf.reduce_sum(probas * labels)
        union = tf.reduce_sum(probas + labels)
        loss = 1 - (2 * inter / union)
        return loss

    @staticmethod
    def _pixel_wise_softmax(labels=None, logits=None, n_classes=2, **kwargs):
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
    segmentation_loss_mixIn
):
    net_structure_class_dict = {
        'UNet': UNetStructure,
        'FusionNet': FusionNetStructure,
    }

    def __init__(self, verbose=10, learning_rate=0.01, learning_rate_decay_rate=0.99,
                 learning_rate_decay_method=None, beta1=0.9, batch_size=100, stage=4,
                 net_type='UNet', loss_type='pixel_wise_softmax', n_classes=2,
                 capacity=64, depth=1, dropout_rate=0.5,
                 **kwargs):
        BaseModel.__init__(self, verbose, **kwargs)
        Xs_MixIn.__init__(self)
        Ys_MixIn.__init__(self)
        supervised_trainMethodMixIn.__init__(self, None)
        predictMethodMixIn.__init__(self)
        predict_probaMethodMixIn.__init__(self)
        scoreMethodMixIn.__init__(self)
        supervised_metricMethodMixIn.__init__(self)
        segmentation_loss_mixIn.__init__(self)

        self.learning_rate = learning_rate
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.learning_rate_decay_method = learning_rate_decay_method
        self.beta1 = beta1
        self.batch_size = batch_size
        self.stage = stage
        self.net_type = net_type
        self.loss_type = loss_type
        self.n_classes = n_classes
        self.capacity = capacity
        self.depth = depth
        self.dropout_rate = dropout_rate

        self.net_structure_class = self.net_structure_class_dict[net_type]

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

        self.net_structure = self.net_structure_class(
            self.Xs, capacity=self.capacity, depth=self.depth, level=self.stage,
            n_classes=self.n_classes
        )
        self.net_structure.build()
        self.vars = self.net_structure.vars
        self._logit = self.net_structure.logit
        self._proba = self.net_structure.proba
        self._predict = reshape(tf.argmax(self._proba, 3, name="predicted"), self.Ys_shape,
                                name='predict')

        self._predict_proba_ops = self._proba
        self._predict_ops = self._predict

    def _build_loss_function(self):
        if self.loss_type == 'BCE+dice_soft':
            self.dice_soft = self._build_loss(
                'dice_soft', labels=self.Ys, logits=self._logit, probas=self._proba,
                predicts=self._predict)
            self.pixel_wise_softmax = self._build_loss(
                'pixel_wise_softmax', labels=self.Ys, logits=self._logit, probas=self._proba,
                predicts=self._predict)

            self.loss = self.dice_soft + self.pixel_wise_softmax
        else:
            self.loss = self._build_loss(
                self.loss_type, labels=self.Ys, logits=self._logit, probas=self._proba, predicts=self._predict)

    def _build_train_ops(self):
        self.drl = TFDynamicLearningRate(self.learning_rate)
        self.drl.build()

        self._train_ops = tf.train.AdamOptimizer(self.drl.learning_rate, beta1=self.beta1) \
            .minimize(self.loss, var_list=self.vars)

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
        self.net_structure.set_train(self.sess)

        Xs, Ys = dataset.next_batch(batch_size, balanced_class=False)
        self.sess.run(self.train_ops, feed_dict={self._Xs: Xs, self._Ys: Ys})

        self.net_structure.set_predict(self.sess)
