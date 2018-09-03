from tqdm import trange
import numpy as np
from script.data_handler.Base.BaseDataset import BaseDataset
from script.model.sklearn_like_model.BaseModel import BaseModel
from script.model.sklearn_like_model.Mixin import Xs_MixIn, Ys_MixIn, supervised_trainMethodMixIn, predictMethodMixIn, \
    predict_probaMethodMixIn, scoreMethodMixIn, supervised_metricMethodMixIn
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


def pixel_wise_softmax(output_map, name='pixel_wise_softmax', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
        return exponential_map / normalize


class UNet(
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

    def __init__(self, verbose=10, learning_rate=0.001, learning_rate_decay_rate=0.99,
                 learning_rate_decay_method=None, beta1=0.01, batch_size=100, stage=4,
                 loss_type='pixel_wise_softmax', n_classes=1, **kwargs):
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
        self.learning_rate_decay_method = learning_rate_decay_method
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.beta1 = beta1
        self.batch_size = batch_size
        self.stage = stage
        self.loss_type = loss_type
        self.n_classes = n_classes

    def _build_input_shapes(self, shapes):
        ret = {}
        ret.update(self._build_Xs_input_shape(shapes))
        ret.update(self._build_Ys_input_shape(shapes))
        return ret

    def _build_main_graph(self):
        self.Xs = tf.placeholder(tf.float32, self.Xs_shape, name='Xs')
        self.Ys = tf.placeholder(tf.float32, self.Ys_shape, name='Ys')

        self.Unet_structure = UNetStructure(self.Xs, level=self.stage)
        self.Unet_structure.build()
        self._logit = self.Unet_structure.logit
        self._proba = self.Unet_structure.proba
        self._predict = reshape(tf.argmax(self._proba, 3, name="predicted"), self.Ys_shape, name='predict')

        self._predict_ops = self._predict
        self.Unet_vars = self.Unet_structure.get_vars()

    def _build_loss_function(self):
        self.loss = self._build_loss(
            self.loss_type, labels=self.Ys, logits=self._logit, probas=self._proba,
            predicts=self._predict)

    def _build_train_ops(self):
        self._train_ops = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
            .minimize(self.loss, var_list=self.Unet_vars)

    @property
    def train_ops(self):
        return self._train_ops

    @property
    def predict_ops(self):
        return self._predict_ops

    @property
    def predict_proba_ops(self):
        return self._proba

    @property
    def score_ops(self):
        return self.metric_ops

    @property
    def metric_ops(self):
        return self.loss

    def train(self, Xs, Ys, epoch=1, save_interval=None, batch_size=None):
        self._prepare_train(Xs=Xs, Ys=Ys)
        dataset = BaseDataset(x=Xs, y=Ys)

        if batch_size is None:
            batch_size = self.batch_size

        iter_num = 0
        iter_per_epoch = int(dataset.size / batch_size)
        if epoch == 1:
            for _ in trange(iter_per_epoch):
                iter_num += 1

                Xs, Ys = dataset.next_batch(batch_size, balanced_class=False)
                self.sess.run(self.train_ops, feed_dict={self._Xs: Xs, self._Ys: Ys})

            Xs, Ys = dataset.next_batch(batch_size, update_cursor=False, balanced_class=False)
            loss = self.sess.run(self.metric_ops, feed_dict={self._Xs: Xs, self._Ys: Ys})
            self.log.info(f"i:{iter_num} loss : {np.mean(loss)}")

            if save_interval is not None:
                self.save()
        else:
            for e in trange(epoch):
                for i in range(iter_per_epoch):
                    iter_num += 1

                    Xs, Ys = dataset.next_batch(batch_size, balanced_class=False)
                    self.sess.run(self.train_ops, feed_dict={self._Xs: Xs, self._Ys: Ys})

                Xs, Ys = dataset.next_batch(batch_size, update_cursor=False, balanced_class=False)
                loss = self.sess.run(self.metric_ops, feed_dict={self._Xs: Xs, self._Ys: Ys})
                self.log.info(f"e:{e}, i:{iter_num} loss : {np.mean(loss)}")

                if save_interval is not None and e % save_interval == 0:
                    self.save()

    def train_early_stop(self, x, y, n_epoch=200, patience=20, min_best=True):
        if min_best is False:
            raise NotImplementedError

        last_metric = np.Inf
        patience_count = 0
        for e in range(1, n_epoch + 1):
            self.train(x, y, epoch=1)
            metric = self.metric(x, y)
            self.log.info(f'e = {e}, metric = {metric}, best = {last_metric}')

            if last_metric > metric:
                self.log.info(f'improve {last_metric - metric}')
                last_metric = metric
                patience_count = 0
            else:
                patience_count += 1

            if patience_count == patience:
                self.log.info(f'early stop')
                break
