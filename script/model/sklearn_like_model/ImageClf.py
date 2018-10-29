from script.model.sklearn_like_model.BaseModel import BaseModel
from script.model.sklearn_like_model.NetModule.InceptionSructure.InceptionV1Structure import InceptionV1NetModule
from script.model.sklearn_like_model.NetModule.InceptionSructure.InceptionV2Structure import InceptionV2NetModule
from script.model.sklearn_like_model.NetModule.InceptionSructure.InceptionV4Structure import InceptionV4NetModule
from script.model.sklearn_like_model.NetModule.MLPNetModule import MLPNetModule
from script.model.sklearn_like_model.NetModule.PlaceHolderModule import PlaceHolderModule
from script.model.sklearn_like_model.NetModule.ResNetStructure.ResNet101NetModule import ResNet101Structure
from script.model.sklearn_like_model.NetModule.ResNetStructure.ResNet152NetModule import ResNet152Structure
from script.model.sklearn_like_model.NetModule.ResNetStructure.ResNet18NetModule import ResNet18NetModule
from script.model.sklearn_like_model.NetModule.ResNetStructure.ResNet34NetModule import ResNet34NetModule
from script.model.sklearn_like_model.NetModule.ResNetStructure.ResNet50NetModule import ResNet50Structure
from script.model.sklearn_like_model.NetModule.TFNormalize import TFL1Normalize, TFL2Normalize
from script.model.sklearn_like_model.NetModule.VGG16NetModule import VGG16NetModule
from script.model.sklearn_like_model.NetModule.optimizer.Adam import Adam
from script.util.tensor_ops import *


class ImageClf(
    BaseModel,
):
    net_structure_class_dict = {
        'VGG16': VGG16NetModule,
        'ResNet18': ResNet18NetModule,
        'ResNet34': ResNet34NetModule,
        'ResNet50': ResNet50Structure,
        'ResNet101': ResNet101Structure,
        'ResNet152': ResNet152Structure,
        'InceptionV1': InceptionV1NetModule,
        'InceptionV2': InceptionV2NetModule,
        'InceptionV4': InceptionV4NetModule,
    }

    def __init__(
            self,
            verbose=10,
            learning_rate=0.01,
            beta1=0.9,
            batch_size=100,
            net_type='VGG16',
            n_classes=2,
            capacity=64,
            use_l1_norm=False,
            l1_norm_rate=0.01,
            use_l2_norm=False,
            l2_norm_rate=0.01,
            dropout_rate=0.5,
            fc_depth=2,
            fc_capacity=1024,
            **kwargs
    ):
        BaseModel.__init__(self, verbose, **kwargs)

        self.n_classes = n_classes
        self.net_type = net_type
        self.batch_size = batch_size
        self.beta1 = beta1
        self.learning_rate = learning_rate
        self.capacity = capacity
        self.use_l1_norm = use_l1_norm
        self.l1_norm_rate = l1_norm_rate
        self.use_l2_norm = use_l2_norm
        self.l2_norm_rate = l2_norm_rate
        self.dropout_rate = dropout_rate
        self.fc_depth = fc_depth
        self.fc_capacity = fc_capacity

    def _build_input_shapes(self, shapes):
        self.x_module = PlaceHolderModule(shapes['x'], name='x').build()
        self.y_module = PlaceHolderModule(shapes['y'], name='y').build()

    def _build_main_graph(self):
        self.Xs = self.x_module.placeholder
        self.Ys = self.y_module.placeholder
        if self.n_classes is None:
            self.n_classes = self.Ys.shape[1]
        self.Ys_label = onehot_to_index(self.Ys)

        net_class = self.net_structure_class_dict[self.net_type]
        self.net_module = net_class(
            self.Xs,
            self.n_classes,
            capacity=self.capacity,
        )
        self.net_module.build()

        self.mlp_net_module = MLPNetModule(
            self.net_module.flatten_layer,
            self.n_classes,
            capacity=self.fc_capacity,
            dropout_rate=self.dropout_rate,
            depth=self.fc_depth,
        ).build()

        self._logit = self.mlp_net_module.logit
        self._proba = self.mlp_net_module.proba
        self.vars = self.net_module.vars

        self._predict = tf.cast(tf.argmax(self._proba, 1, name="predicted_label"), tf.float32)

        self._predict_proba_ops = self._proba
        self._predict_ops = self._predict

        self.acc = tf.cast(tf.equal(self._predict, self.Ys_label), tf.float64, name="acc")
        self.acc_mean = tf.reduce_mean(self.acc, name="acc_mean")

    def _build_loss_ops(self):
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Ys, logits=self._logit)

        if self.use_l1_norm:
            self.l1_norm = TFL1Normalize(self.net_module.vars, self.l1_norm_rate)
            self.l1_norm.build()
            self.loss += self.l1_norm.penalty

        if self.use_l2_norm:
            self.l2_norm = TFL2Normalize(self.net_module.vars, self.l1_norm_rate)
            self.l2_norm.build()
            self.loss += self.l2_norm.penalty

        # TODO
        # self.loss = self.loss + self.l1_norm_penalty
        # # average top k loss
        # self.loss = average_top_k_loss(self.loss, self.K_average_top_k_loss)

        self._metric_ops = self.loss

    def _build_train_ops(self):
        self.optimizer = Adam(self.loss, self.vars, self.learning_rate).build()
        self.train_ops = self.optimizer.train_op

    def _train_iter(self, dataset, batch_size):
        self.set_train(self.sess)
        Xs, Ys = dataset.next_batch(batch_size, balanced_class=False)
        self.sess.run(self.train_ops, feed_dict={self.Xs: Xs, self.Ys: Ys})
        self.set_non_train(self.sess)

    def _metric(self, x=None, y=None):
        return self.batch_execute(self.loss, {self.Xs: x, self.Ys: y})

    def update_learning_rate(self, lr):
        if self.sess is not None:
            self.optimizer.update_learning_rate(self.sess, self.learning_rate)

    def set_train(self, sess):
        self.mlp_net_module.set_train(sess)

    def set_non_train(self, sess):
        self.mlp_net_module.set_non_train(sess)

    def init_adam_momentum(self):
        self.sess.run(tf.variables_initializer(self.train_ops_var_list))

