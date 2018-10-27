from pprint import pformat

import numpy as np
import tensorflow as tf
from tqdm import trange, tqdm

from env_settting import *
from script.data_handler.Base.BaseDataset import BaseDataset
from script.model.sklearn_like_model.Mixin import paramsMixIn, loss_packMixIn, slice_np_arr
from script.model.sklearn_like_model.NetModule.PlaceHolderModule import PlaceHolderModule
from script.model.sklearn_like_model.SessionManager import SessionManager
from script.model.sklearn_like_model.callback.BaseEpochCallback import BaseEpochCallback
from script.util.MixIn import LoggerMixIn
from script.util.misc_util import setup_directory, setup_file, dump_json, load_json
from script.util.misc_util import time_stamp, path_join, error_trace
from script.util.tensor_ops import join_scope


class ModelBuildFailError(BaseException):
    pass


class TrainFailError(BaseException):
    pass


class BaseDatasetCallback:
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def shuffle(self):
        raise NotImplementedError

    def next_batch(self, batch_size, batch_keys=None, update_cursor=True, balanced_class=False, out_type='concat'):
        raise NotImplementedError

    @property
    def size(self):
        raise NotImplementedError


class BaseDataCollector(BaseEpochCallback):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)

    def update_data(self, model, dataset, metric, epoch):
        raise NotImplementedError

    def __call__(self, model, dataset, metric, epoch):
        self.update_data(model, dataset, metric, epoch)


META_DATA_FILE_NAME = 'instance.meta'
meta_json = 'meta.json'
params_json = 'params.json'
input_shapes_json = 'input_shapes.json'
INSTANCE_FOLDER = 'instance'


class ModelMetadata:
    def __init__(self, id_, **kwargs):
        self.id = id_
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.keys = ['id'] + list(kwargs.keys())

    def __str__(self):
        return pformat(self.metadata)

    def __repr__(self):
        return self.__str__()

    @property
    def metadata(self):
        return {
            key: getattr(self, key, None)
            for key in self.keys
        }

    def load(self, path):
        obj = load_json(path)
        for key, val in obj.items():
            setattr(self, key, val)

    def save(self, path):
        dump_json(self.metadata, path)


class BaseModel(
    LoggerMixIn,
    paramsMixIn,
    loss_packMixIn
):
    """Abstract class of model for tensorflow graph"""
    AUTHOR = 'demetoir'

    def __init__(self, verbose=10, **kwargs):
        """create instance of AbstractModel

        :param verbose:
        :type logger_path: str
        :param logger_path: path for log file
        if logger_path is None, log ony stdout
        """
        LoggerMixIn.__init__(self, verbose=verbose)
        paramsMixIn.__init__(self)
        loss_packMixIn.__init__(self)

        self.sessionManager = SessionManager(
            sess=kwargs['sess'] if 'sess' in kwargs else None,
            config=kwargs['config'] if 'config' in kwargs else None,
        )

        self._is_input_shape_built = False
        self._is_graph_built = False

        self.verbose = verbose
        # gen instance id

        if 'run_id' in kwargs:
            self.run_id = kwargs['run_id']
        else:
            self.run_id = time_stamp()

        if 'id' in kwargs:
            id_ = kwargs['id']
        else:
            id_ = "_".join(["%s_%s" % (self.AUTHOR, self.__class__.__name__), self.run_id])
        self.metadata = ModelMetadata(
            id_=id_,
        )

    @property
    def id(self):
        return self.metadata.id

    def __str__(self):
        s = ""
        s += "%s_%s\n" % (self.AUTHOR, self.__class__.__name__)
        s += pformat({
            'id': self.id,
            'run_id': self.run_id,
            'params': self.params,
            'meta': self.metadata

        })
        return s

    def __repr__(self):
        s = ""
        s += "%s_%s" % (self.AUTHOR, self.__class__.__name__)
        s += pformat(self.params)
        return s

    def __del__(self):
        del self.sessionManager

    @property
    def sess(self):
        return self.sessionManager.sess

    @property
    def var_list(self):
        return tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=self.metadata.id
        )

    @property
    def main_graph_var_list(self):
        return tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=join_scope(self.metadata.id, 'main_graph')
        )

    @property
    def train_ops_var_list(self):
        return tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=join_scope(self.metadata.id, 'train_ops')
        )

    @property
    def misc_ops_var_list(self):
        return tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=join_scope(self.metadata.id, 'misc_ops')
        )

    @property
    def loss_ops_var_list(self):
        return tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=join_scope(self.metadata.id, 'loss_ops')
        )

    @property
    def trainable_var_list(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.metadata.id)

    @property
    def is_built(self):
        return self._is_input_shape_built and self._is_graph_built

    def _build_graph(self):
        try:
            with tf.variable_scope(str(self.metadata.id)):
                with tf.variable_scope("misc_ops"):
                    self.log.debug("build_misc_ops")
                    self._build_misc_ops()

                with tf.variable_scope('main_graph'):
                    self.log.debug('build_main_graph')
                    self._build_main_graph()

                with tf.variable_scope('loss_ops'):
                    self.log.debug('build_loss_ops')
                    self._loss_ops = self._build_loss_ops()
                    assert self._loss_ops is not None

                with tf.variable_scope('predict_ops'):
                    self.log.debug('build_predict_ops')
                    self._predict_ops = self._build_predict_ops()
                    assert self._predict_ops is not None

                with tf.variable_scope('metric_ops'):
                    self.log.debug('build_metric_ops')
                    self._metric_ops = self._build_metric_ops()
                    assert self._metric_ops is not None

                with tf.variable_scope('train_ops'):
                    self.log.debug('build_train_ops')
                    self._train_ops = self._build_train_ops()
                    assert self._train_ops is not None

            self._is_graph_built = True
            self.log.info("build success")

        except BaseException as e:
            self.log.error(error_trace(e))
            print(error_trace(e))
            raise ModelBuildFailError("ModelBuildFailError")

    def _build_input_shapes(self, x=None, y=None):
        self.input_modules = {}

        if type(x) is dict:
            raise NotImplementedError
        elif type(x) is tuple:
            self.x_module = PlaceHolderModule(x, name='x').build()
            self.Xs = self.x_module.placeholder
            self.input_modules['x'] = self.x_module

        else:
            raise ValueError(f'x expect tuple or dict, but x = {x}')

        if type(y) is dict:
            raise NotImplementedError
        elif type(y) is tuple:
            self.y_module = PlaceHolderModule(y, name='y').build()
            self.Ys = self.y_module.placeholder
            self.input_modules['y'] = self.y_module
        else:
            raise ValueError(f'y expect tuple or dict, but y = {y}')

    def get_inputs_module(self, name):
        return self.input_modules[name]

    def _build_main_graph(self):
        """load main tensor graph

        :raise NotImplementError
        if not implemented
        """
        raise NotImplementedError

    def _build_loss_ops(self):
        """load loss function of model

        :raise NotImplementError
        if not implemented
        """
        raise NotImplementedError

    def _build_misc_ops(self):
        """load misc operation of model

        :raise NotImplementError
        if not implemented
        """
        init = tf.constant(0, dtype=tf.int32)
        self.global_step = tf.get_variable("global_step", initializer=init)
        self.op_inc_global_step = tf.assign_add(self.global_step, 1, name='op_inc_global_step')

        self.global_epoch = tf.get_variable("global_epoch", initializer=init)
        self.op_inc_global_epoch = tf.assign_add(self.global_epoch, 1, name='op_inc_global_epoch')

    def _build_train_ops(self):
        """Load train operation of model

        :raise NotImplementError
        if not implemented
        """
        raise NotImplementedError

    def _build_metric_ops(self):
        raise NotImplementedError

    def _build_predict_ops(self):
        raise NotImplementedError

    def build(self, x=None, y=None):
        if not self._is_input_shape_built:
            try:
                self.inputs_x = x
                self.inputs_y = y
                self._build_input_shapes(x, y)
                self._is_input_shape_built = True
            except BaseException as e:
                print(error_trace(e))
                raise ModelBuildFailError(f'input_shape build fail, x={x}, y={y}')

        self._build_graph()

        self.sessionManager.open_if_not()
        self.sessionManager.init_variable(self.var_list)

    def save_meta(self, path):
        setup_directory(path)

        self.metadata.save(path_join(path, 'meta.pkl'))
        self.metadata.save(path_join(path, 'meta.json'))

        self.params_path = path_join(path, 'params.pkl')
        self._save_params(self.params_path)

    def save_checkpoint(self, path):
        setup_directory(path)

        check_point_path = path_join(path, 'check_point', 'instance.ckpt')
        setup_file(check_point_path)
        saver = tf.train.Saver(self.var_list)
        saver.save(self.sess, check_point_path)

    def save(self, path=None):
        if not self.is_built:
            raise RuntimeError(f'can not save un built model, {self}')

        if not self.sessionManager.is_opened:
            raise RuntimeError(f'can not save model without session {self}')

        if path is None:
            self.log.info('save directory not specified, use default directory')
            path = path_join(ROOT_PATH, 'instance', self.metadata.id)

        self.log.info("save at {}".format(path))
        self.save_checkpoint(path)
        self.save_meta(path)

    def load_meta(self, path):
        self.log.info(f'load from {path}')
        self.metadata.load(path_join(path, 'meta.json'))
        self._load_params(path_join(path, 'params.pkl'))

        return self

    def restore(self, path, var_list=None):
        self.log.info(f'restore from {path}')

        if var_list is None:
            var_list = self.main_graph_var_list + self.misc_ops_var_list

        saver = tf.train.Saver(var_list)
        saver.restore(self.sess, path_join(path, 'check_point', 'instance.ckpt'))

    def reset_global_epoch(self):
        self.sess.run(tf.initialize_variables([self.global_step]))

    def batch_execute(self, op, inputs):
        batch_size = getattr(self, 'batch_size', 32)
        size = 0
        for val in inputs.values():
            size = len(val)

        if size > batch_size:
            partial = {}
            for key, val in inputs.items():
                for idx, part in enumerate(slice_np_arr(val, batch_size)):
                    if idx not in partial:
                        partial[idx] = {}

                    partial[idx][key] = part

            partial = list(partial.values())

            tqdm.write('batch predict')
            batchs = [
                self.sess.run(op, feed_dict=x_partial)
                for x_partial in tqdm(partial)
            ]

            try:
                return np.concatenate(batchs)
            except:
                return batchs
        else:
            return self.sess.run(op, feed_dict=inputs)

    def _loss_check(self, loss_pack):
        for key, item, in loss_pack.items():
            if any(np.isnan(item)):
                self.log.error(f'{key} is nan')
                raise TrainFailError(f'{key} is nan')
            if any(np.isinf(item)):
                self.log.error(f'{key} is inf')
                raise TrainFailError(f'{key} is inf')

    def _is_fine_metric(self, metric):
        if metric in (np.nan, np.inf, -np.inf):
            print('metric is {metric}')
            return True

        if metric == getattr(self, 'recent_metric', None):
            return True
        else:
            setattr(self, 'recent_metric', metric)

        return False

    def train_iter(self, x=None, y=None, batch_size=None):
        self._train_iter(BaseDataset(x=x, y=y), batch_size)

    def train(
            self, x=None, y=None, epoch=1, batch_size=None,
            dataset_callback=None, epoch_pbar=True, iter_pbar=True, epoch_callbacks=None,
    ):

        if not self.is_built:
            raise RuntimeError(f'{self} not built')

        batch_size = getattr(self, 'batch_size') if batch_size is None else batch_size
        self.batch_size = batch_size
        dataset = dataset_callback if dataset_callback else BaseDataset(x=x, y=y)

        metric = None
        epoch_pbar = tqdm([i for i in range(1, epoch + 1)]) if epoch_pbar else None
        for _ in range(1, epoch + 1):
            dataset.shuffle()

            iter_pbar = trange if iter_pbar else range
            for _ in iter_pbar(int(dataset.size / batch_size)):
                self._train_iter(dataset, batch_size)

            self.sess.run(self.op_inc_global_epoch)
            global_epoch = self.sess.run(self.global_epoch)
            if epoch_pbar: epoch_pbar.update(1)

            metric = self.metric(x, y)
            loss = self.loss(x, y)
            tqdm.write(f"epoch:{global_epoch}, loss={loss}, metric={np.mean(metric)}\n")
            if self._is_fine_metric(metric):
                break

            break_epoch = False
            if epoch_callbacks:
                try:
                    results = [
                        callback(self, dataset, metric, global_epoch)
                        for callback in epoch_callbacks
                    ]
                except BaseException as e:
                    print(error_trace(e))
                    raise RuntimeError
                for result in results:
                    if result and 'break_epoch' in result:
                        break_epoch = True
            if break_epoch: break

        if epoch_pbar: epoch_pbar.close()
        if dataset_callback: del dataset

        return metric

    def metric(self, x=None, y=None, mean=True):
        metric = self.batch_execute(self._metric_ops, {self.Xs: x, self.Ys: y})
        if mean:
            metric = np.mean(metric)
        return metric

    def loss(self, x=None, y=None, mean=True):
        loss = self.batch_execute(self._loss_ops, {self.Xs: x, self.Ys: y})
        if mean:
            loss = np.mean(loss)
        return loss

    def predict(self, x=None, y=None):
        return self.batch_execute(self._predict_ops, {self.Xs: x, self.Ys: y})

    def evaluate(self, x=None, y=None):
        return self.batch_execute(self._metric_ops, {self.Xs: x, self.Ys: y})

    def _train_iter(self, dataset, batch_size):
        Xs, Ys = dataset.next_batch(batch_size)
        self.sess.run(self._train_ops, feed_dict={self.Xs: Xs, self.Ys: Ys})
