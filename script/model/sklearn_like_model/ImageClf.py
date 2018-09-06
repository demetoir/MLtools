from script.model.sklearn_like_model.BaseModel import BaseModel
from script.model.sklearn_like_model.Mixin import Xs_MixIn, Ys_MixIn, supervised_trainMethodMixIn, predictMethodMixIn, \
    predict_probaMethodMixIn, scoreMethodMixIn, supervised_metricMethodMixIn
from script.model.sklearn_like_model.net_structure.ResNetStructure import ResNetStructure
from script.model.sklearn_like_model.net_structure.VGG16Structure import VGG16Structure
from script.util.Stacker import Stacker
from script.util.tensor_ops import *


class ImageClf(
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
        'VGG': VGG16Structure,
        'ResNet': ResNetStructure,
    }

    def __init__(self, verbose=10, learning_rate=0.01, learning_rate_decay_rate=0.99,
                 learning_rate_decay_method=None, beta1=0.01, batch_size=100, net_type='VGG',
                 net_structure_args=None, net_structure_kwargs=None, net_structure_class=None, n_classes=None,
                 **kwargs):
        BaseModel.__init__(self, verbose, **kwargs)
        Xs_MixIn.__init__(self)
        Ys_MixIn.__init__(self)
        supervised_trainMethodMixIn.__init__(self, None)
        predictMethodMixIn.__init__(self)
        predict_probaMethodMixIn.__init__(self)
        scoreMethodMixIn.__init__(self)
        supervised_metricMethodMixIn.__init__(self)

        self.n_classes = n_classes
        self.net_structure_kwargs = net_structure_kwargs
        self.net_structure_args = net_structure_args
        self.net_type = net_type
        self.batch_size = batch_size
        self.beta1 = beta1
        self.learning_rate_decay_method = learning_rate_decay_method
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.learning_rate = learning_rate

        self.net_structure = None
        if net_structure_class is None:
            self.net_structure_class = self.net_structure_class_dict[self.net_type]
        else:
            self.net_structure_class = net_structure_class

    def _build_input_shapes(self, shapes):
        ret = {}
        ret.update(self._build_Xs_input_shape(shapes))
        ret.update(self._build_Ys_input_shape(shapes))
        return ret

    def classifier(self, Xs, net_shapes, name='classifier'):
        with tf.variable_scope(name):
            layer = Stacker(flatten(Xs))

            for net_shape in net_shapes:
                layer.linear_block(net_shape, relu)

            layer.linear(self.Y_flatten_size)
            logit = layer.last_layer
            h = softmax(logit)
        return logit, h

    def _build_main_graph(self):
        self.Xs = tf.placeholder(tf.float32, self.Xs_shape, name='Xs')
        self.Ys = tf.placeholder(tf.float32, self.Ys_shape, name='Ys')
        if self.n_classes is None:
            self.n_classes = self.Ys.shape[1]
        self.Ys_label = onehot_to_index(self.Ys)

        self.net_structure = self.net_structure_class(
            self.Xs, self.n_classes,
            # todo add structure args kwargs
            # *self.net_structure_args, **self.net_structure_kwargs
        )
        self.net_structure.build()
        self._logit = self.net_structure.logit
        self._proba = self.net_structure.proba
        self.vars = self.net_structure.vars

        self._predict = tf.cast(tf.argmax(self._proba, 1, name="predicted_label"), tf.float32)

        self._predict_proba_ops = self._proba
        self._predict_ops = self._predict

        self.acc = tf.cast(tf.equal(self._predict, self.Ys_label), tf.float64, name="acc")
        self.acc_mean = tf.reduce_mean(self.acc, name="acc_mean")

    def _build_loss_function(self):
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Ys, logits=self._logit)

        # todo normalize term
        # self.l1_norm_penalty = L1_norm(self.vars, lambda_=self.l1_norm_lambda)
        # self.l1_norm_penalty_mean = tf.reduce_mean(self.l1_norm_penalty, name='l1_norm_penalty_mean')
        # # self.l1_norm_penalty *= wall_decay(0.999, self.global_step, 100)
        # self.l2_norm_penalty = L2_norm(self.vars, lambda_=self.l2_norm_lambda)
        # self.l2_norm_penalty_mean = tf.reduce_mean(self.l2_norm_penalty, name='l2_norm_penalty_mean')
        #
        # self.loss = self.loss + self.l1_norm_penalty
        # # average top k loss
        # self.loss = average_top_k_loss(self.loss, self.K_average_top_k_loss)
        self._metric_ops = self.loss

    def _build_train_ops(self):
        self._train_ops = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss, var_list=self.vars)

    def _train_iter(self, dataset, batch_size):
        Xs, Ys = dataset.next_batch(batch_size, balanced_class=False)
        self.sess.run(self.train_ops, feed_dict={self._Xs: Xs, self._Ys: Ys})

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
        return self.acc_mean

    @property
    def metric_ops(self):
        return self._metric_ops
