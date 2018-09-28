from pprint import pprint

from script.model.sklearn_like_model.TFNormalize import TFL1Normalize, TFL2Normalize
from script.model.sklearn_like_model.BaseModel import BaseEpochCallback
from script.model.sklearn_like_model.BaseModel import BaseModel
from script.model.sklearn_like_model.Mixin import Xs_MixIn, Ys_MixIn, supervised_trainMethodMixIn, predictMethodMixIn, \
    predict_probaMethodMixIn, scoreMethodMixIn, supervised_metricMethodMixIn
from script.model.sklearn_like_model.TFSummary import TFSummaryScalar
from script.model.sklearn_like_model.callback.Top_k_save import Top_k_save
from script.model.sklearn_like_model.NetModule.InceptionSructure.InceptionV1Structure import InceptionV1NetModule
from script.model.sklearn_like_model.NetModule.InceptionSructure.InceptionV2Structure import InceptionV2NetModule
from script.model.sklearn_like_model.NetModule.InceptionSructure.InceptionV4Structure import InceptionV4NetModule
from script.model.sklearn_like_model.NetModule.ResNetStructure.ResNet101NetModule import ResNet101Structure
from script.model.sklearn_like_model.NetModule.ResNetStructure.ResNet152NetModule import ResNet152Structure
from script.model.sklearn_like_model.NetModule.ResNetStructure.ResNet18NetModule import ResNet18NetModule
from script.model.sklearn_like_model.NetModule.ResNetStructure.ResNet34NetModule import ResNet34NetModule
from script.model.sklearn_like_model.NetModule.ResNetStructure.ResNet50NetModule import ResNet50Structure
from script.model.sklearn_like_model.NetModule.VGG16NetModule import VGG16NetModule
from script.util.misc_util import time_stamp, path_join
from script.util.numpy_utils import *
from script.util.tensor_ops import *
from script.workbench.TGS_salt.TGS_salt_inference import TGS_salt_DataHelper, plot, to_dict, save_tf_summary_params

task_name = 'mask_rate_reg'
SUMMARY_PATH = f'./tf_summary/TGS_salt/mask_rate_reg'
INSTANCE_PATH = f'./instance/TGS_salt/mask_rate_reg'
PLOT_PATH = f'./matplot/TGS_salt/mask_rate_reg'


class MaskRateReg(
    BaseModel,
    Xs_MixIn,
    Ys_MixIn,
    supervised_trainMethodMixIn,
    predictMethodMixIn,
    scoreMethodMixIn,
    supervised_metricMethodMixIn,
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
    loss_funcs = {
        'MSE': MSE_loss,
        'RMSE': RMSE_loss,
        'MAE': MAE_loss,
    }

    def __init__(self, verbose=10, learning_rate=0.01, learning_rate_decay_rate=0.99,
                 learning_rate_decay_method=None, beta1=0.9, batch_size=100,
                 net_type='VGG16', capacity=64, loss_type='MSE',
                 use_l1_norm=False, l1_norm_rate=0.01,
                 use_l2_norm=False, l2_norm_rate=0.01,
                 **kwargs):
        BaseModel.__init__(self, verbose, **kwargs)
        Xs_MixIn.__init__(self)
        Ys_MixIn.__init__(self)
        supervised_trainMethodMixIn.__init__(self, None)
        predictMethodMixIn.__init__(self)
        predict_probaMethodMixIn.__init__(self)
        scoreMethodMixIn.__init__(self)
        supervised_metricMethodMixIn.__init__(self)

        self.net_type = net_type
        self.batch_size = batch_size
        self.beta1 = beta1
        self.learning_rate_decay_method = learning_rate_decay_method
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.learning_rate = learning_rate
        self.net_capacity = capacity
        self.loss_type = loss_type

        self.use_l1_norm = use_l1_norm
        self.l1_norm_rate = l1_norm_rate
        self.use_l2_norm = use_l2_norm
        self.l2_norm_rate = l2_norm_rate

        self.net_structure = None
        self.net_structure_class = self.net_structure_class_dict[self.net_type]

    def _build_input_shapes(self, shapes):
        ret = {}
        ret.update(self._build_Xs_input_shape(shapes))
        ret.update(self._build_Ys_input_shape(shapes))
        return ret

    def _build_main_graph(self):
        self.Xs = tf.placeholder(tf.float32, self.Xs_shape, name='Xs')
        self.Ys = tf.placeholder(tf.float32, self.Ys_shape, name='Ys')

        self.net_structure = self.net_structure_class(
            self.Xs, n_classes=1, capacity=self.net_capacity,
        )
        self.net_structure.build()
        self._predict = self.net_structure.logit
        self.vars = self.net_structure.vars

        self._predict_ops = self._predict

    def _build_loss_function(self):
        self.loss = self.loss_funcs[self.loss_type](self._predict, self.Ys)

        if self.use_l1_norm:
            self.l1_norm = TFL1Normalize(self.net_structure.vars, self.l1_norm_rate)
            self.l1_norm.build()
            self.loss += self.l1_norm.penalty

        if self.use_l2_norm:
            self.l2_norm = TFL2Normalize(self.net_structure.vars, self.l1_norm_rate)
            self.l2_norm.build()
            self.loss += self.l2_norm.penalty

        # TODO
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
    def score_ops(self):
        return self._metric_ops

    @property
    def metric_ops(self):
        return self._metric_ops


class EpochCallback(BaseEpochCallback):
    def __init__(self, model, train_x, train_y, test_x, test_y, params):
        super().__init__()
        self.train_x = train_x
        self.train_y = train_y
        self.model = model
        self.test_x = test_x
        self.test_y = test_y
        self.params = params

        self.run_id = self.params['run_id']

        self.top_k_save = Top_k_save(path_join(INSTANCE_PATH, self.run_id, 'top_k'), max_best=False)

        self.summary_train_loss = TFSummaryScalar(path_join(SUMMARY_PATH, self.run_id, 'train'), 'train_loss')
        self.summary_train_acc = TFSummaryScalar(path_join(SUMMARY_PATH, self.run_id, 'train'), 'train_acc')
        self.summary_test_acc = TFSummaryScalar(path_join(SUMMARY_PATH, self.run_id, 'test'), 'test_acc')

        self.sample_size = 200
        self.sample_train_x, self.sample_train_y = self.make_plot_data(self.test_x, self.test_y, self.sample_size)
        self.sample_test_x, self.sample_test_y = self.make_plot_data(self.train_x, self.train_y, self.sample_size)

    def log_score(self, epoch, log):
        self.train_loss = self.model.metric(self.train_x, self.train_y)
        self.test_score = self.model.score(self.test_x, self.test_y)
        self.train_score = self.model.score(self.train_x, self.train_y)

        log(
            f'e={epoch}, '
            f'train_score = {self.train_score},\n'
            f'test_score = {self.test_score},\n'
            f'train_loss = {self.train_loss}\n'
        )

    def update_summary(self, sess, epoch):
        self.summary_train_loss.update(sess, self.train_loss, epoch)
        self.summary_train_acc.update(sess, self.train_score, epoch)
        self.summary_test_acc.update(sess, self.test_score, epoch)

    def make_plot_data(self, xs, ys, size):

        size = 200
        a = []
        for i in range(size):
            x = xs[i]
            y = ys[i]
            a += [(x, y)]
        a = sorted(a, key=lambda x: x[1])

        xs = []
        ys = []
        for x, y in a:
            xs += [x]
            ys += [y]
        xs = np.array(xs)
        ys = np.array(ys)
        return xs, ys

    def save_plot(self, epoch):
        def _plot_scatter(model, x, y, plot, name):
            predict = model.predict(x)
            gt = np.array([[idx, y] for idx, y in enumerate(y)])
            predict = np.array([[idx, x] for idx, x in enumerate(predict)])
            plot.scatter_2d(
                gt, predict, labels=['gt', 'predict'],
                title=f'mask_rate_{name}_{epoch}',
                path=path_join(PLOT_PATH, self.run_id, f'{name}', f'({epoch})'))

        _plot_scatter(self.model, self.sample_train_x, self.sample_train_y, plot, 'train')
        _plot_scatter(self.model, self.sample_test_x, self.sample_test_y, plot, 'test')

    def __call__(self, sess, dataset, epoch, log=None):
        self.log_score(epoch, log)
        self.update_summary(sess, epoch)
        self.top_k_save(float(self.test_score), self.model)
        self.save_plot(epoch)


class mask_rate_reg_pipeline:
    def __init__(self):
        self.data_helper = TGS_salt_DataHelper()
        self.plot = plot

        self.init_dataset()

    def init_dataset(self):
        # train_set = self.data_helper.train_set
        # self.data_helper.train_set.y_keys = ['mask_rate']
        train_set = self.data_helper.train_set_non_empty_mask
        train_set.y_keys = ['mask_rate']

        train_x, train_y = train_set.full_batch()
        train_x = train_x.reshape([-1, 101, 101, 1])
        train_y = train_y.reshape([-1, 1])

        from sklearn.model_selection import train_test_split
        train_x, test_x, train_y, test_y = train_test_split(
            train_x, train_y, test_size=0.33)

        sample_size = 100
        sample_x = train_x[:sample_size]
        sample_y = train_y[:sample_size]

        self.train_set = train_set
        self.train_x = train_x
        self.train_y = train_y * 100
        self.test_x = test_x
        self.test_y = test_y * 100
        self.sample_x = sample_x
        self.sample_y = sample_y * 100

    def params(self, run_id=None,
               learning_rate=0.01, learning_rate_decay_rate=0.99,
               learning_rate_decay_method=None, beta1=0.9, batch_size=100,
               net_type='VGG16', capacity=64, loss_type='MSE',
               use_l1_norm=False, l1_norm_rate=0.01,
               use_l2_norm=False, l2_norm_rate=0.01, comment=None):
        # net_type = 'InceptionV1'
        # net_type = 'InceptionV2'
        # net_type = 'InceptionV4'
        # net_type = 'ResNet18'
        # net_type = 'ResNet34'
        # net_type = 'ResNet50'
        # net_type = 'ResNet101'
        # net_type = 'ResNet152'

        if run_id is None:
            run_id = time_stamp()

        return to_dict(
            run_id=run_id,
            batch_size=batch_size,
            net_type=net_type,
            capacity=capacity,
            learning_rate=learning_rate,
            beta1=beta1,
            loss_type=loss_type,
            use_l1_norm=use_l1_norm,
            l1_norm_rate=l1_norm_rate,
            use_l2_norm=use_l2_norm,
            l2_norm_rate=l2_norm_rate,
            comment=comment
        )

    def train(self, params, n_epoch, augmentation=False, early_stop=True, patience=20):
        save_tf_summary_params(SUMMARY_PATH, params)
        pprint(f'train {params}')

        clf = MaskRateReg(**params)
        epoch_callback = EpochCallback(
            clf,
            self.train_x, self.train_y,
            self.test_x, self.test_y,
            params,
        )

        clf.train(self.train_x, self.train_y, epoch=n_epoch, epoch_callbacks=epoch_callback,
                  iter_pbar=True, dataset_callback=None, early_stop=early_stop, patience=patience)

        score = clf.score(self.sample_x, self.sample_y)
        test_score = clf.score(self.test_x, self.test_y)
        print(f'score = {score}, test= {test_score}')
