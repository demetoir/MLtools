from script.model.sklearn_like_model.BaseClassifierModel import BaseClassifierModel
from script.util.Stacker import Stacker
from script.util.tensor_ops import *

MLP_param_keys = [
    'batch_size',
    'learning_rate',
    'beta1',
    'dropout_rate',
    'K_average_top_k_loss',
    'net_shapes',
    'activation',
    'l1_norm_lambda',
    'l2_norm_lambda'
]


class MLPClassifier(BaseClassifierModel):
    _params_keys = MLP_param_keys

    def __init__(self, verbose=10, batch_size=100, learning_rate=0.01, beta1=0.9, k_average_top_loss=20,
                 net_shapes=(256, 256,), activation='relu', l1_norm_lambda=0.001, l2_norm_lambda=0.001, **kwargs):
        super().__init__(verbose, **kwargs)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.K_average_top_k_loss = k_average_top_loss
        self.net_shapes = net_shapes
        self.activation = activation
        self.l1_norm_lambda = l1_norm_lambda
        self.l2_norm_lambda = l2_norm_lambda

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

        self.logit, self.h = self.classifier(self.Xs, self.net_shapes)

        self.vars = collect_vars(join_scope(get_scope(), 'classifier'))

        self.predict_index = tf.cast(tf.argmax(self.h, 1, name="predicted_label"), tf.float32)
        self.label_index = onehot_to_index(self.Ys)
        self.acc = tf.cast(tf.equal(self.predict_index, self.label_index), tf.float64, name="acc")
        self.acc_mean = tf.reduce_mean(self.acc, name="acc_mean")

    def _build_loss_ops(self):
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Ys, logits=self.logit)

        self.l1_norm_penalty = L1_norm(self.vars, lambda_=self.l1_norm_lambda)
        self.l1_norm_penalty_mean = tf.reduce_mean(self.l1_norm_penalty, name='l1_norm_penalty_mean')
        # self.l1_norm_penalty *= wall_decay(0.999, self.global_step, 100)
        self.l2_norm_penalty = L2_norm(self.vars, lambda_=self.l2_norm_lambda)
        self.l2_norm_penalty_mean = tf.reduce_mean(self.l2_norm_penalty, name='l2_norm_penalty_mean')

        self.loss = self.loss + self.l1_norm_penalty
        # average top k loss
        # self.loss = average_top_k_loss(self.loss, self.K_average_top_k_loss)
        self.loss_mean = tf.reduce_mean(self.loss, name='loss_mean')

    def _build_train_ops(self):
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss, var_list=self.vars)
