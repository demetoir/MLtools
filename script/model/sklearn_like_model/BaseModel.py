from pprint import pformat

import numpy as np
import tensorflow as tf
from tqdm import trange, tqdm

from env_settting import *
from script.data_handler.Base.BaseDataset import BaseDataset
from script.model.sklearn_like_model.Mixin import input_shapesMixIn, paramsMixIn, loss_packMixIn
from script.model.sklearn_like_model.SessionManager import SessionManager
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


class BaseEpochCallback:
    def __call__(self, model, dataset, metric, epoch):
        raise NotImplementedError

    def trace_on(self, dc, key):
        self.dc = dc
        self.dc_key = key

    def hook_metric(self, metric):
        if hasattr(self, 'dc') and hasattr(self, 'dc_key'):
            metric = getattr(self.dc, self.dc_key)
        return metric

    def on_start_epoch(self, model, dataset, metric, epoch):
        pass

    def on_end_epoch(self, model, dataset, metric, epoch):
        pass

    def on_start_train(self, model, dataset, metric, epoch):
        pass

    def on_end_train(self, model, dataset, metric, epoch):
        pass


META_DATA_FILE_NAME = 'instance.meta'
meta_json = 'meta.json'
params_json = 'params.json'
input_shapes_json = 'input_shapes.json'
INSTANCE_FOLDER = 'instance'


class ModelMetadata:
    def __init__(self, id_, run_id, **kwargs):
        self.id = id_
        self.run_id = run_id
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.keys = ['id', 'run_id'] + list(kwargs.keys())

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
    input_shapesMixIn,
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
        input_shapesMixIn.__init__(self)
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
        run_id = kwargs['run_id'] if 'run_id' in kwargs else time_stamp()
        id_ = "_".join(["%s_%s" % (self.AUTHOR, self.__class__.__name__), run_id])
        self.metadata = ModelMetadata(
            id_=id_,
            run_id=run_id
        )

    def __str__(self):
        s = ""
        s += "%s_%s\n" % (self.AUTHOR, self.__class__.__name__)
        s += pformat(self.params)
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

                self.log.debug('build_main_graph')
                with tf.variable_scope('main_graph'):
                    self._build_main_graph()

                with tf.variable_scope('loss_function'):
                    self.log.debug('build_loss_function')
                    self._build_loss_function()

                with tf.variable_scope('train_ops'):
                    self.log.debug('build_train_ops')
                    self._build_train_ops()

            self._is_graph_built = True
            self.log.info("build success")

        except BaseException as e:
            self.log.error(error_trace(e))
            print(error_trace(e))
            raise ModelBuildFailError("ModelBuildFailError")

    def _build_input_shapes(self, shapes):
        """load input shapes for tensor placeholder

        :type shapes: dict
        :param shapes: input shapes for tensor placeholder

        :raise NotImplementError
        if not Implemented
        """
        raise NotImplementedError

    def _build_main_graph(self):
        """load main tensor graph

        :raise NotImplementError
        if not implemented
        """
        raise NotImplementedError

    def _build_loss_function(self):
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

    def build(self, **inputs):
        if not self._is_input_shape_built:
            try:
                self.inputs = inputs
                input_shapes = self._build_input_shapes(inputs)
                self._apply_input_shapes(input_shapes)
                self._is_input_shape_built = True
            except BaseException as e:
                print(error_trace(e))
                raise ModelBuildFailError(f'input_shape build fail, {inputs}')

        self._build_graph()

        self.sessionManager.open_if_not()
        self.sessionManager.init_variable(self.var_list)

    def save(self, path=None):
        if not self.is_built:
            raise RuntimeError(f'can not save un built model, {self}')

        if not self.sessionManager.is_opened:
            raise RuntimeError(f'can not save model without session {self}')

        if path is None:
            self.log.info('save directory not specified, use default directory')
            path = path_join(ROOT_PATH, 'instance', self.metadata.id)

        self.log.info("saved at {}".format(path))
        setup_directory(path)

        check_point_path = path_join(path, 'check_point', 'instance.ckpt')
        setup_file(check_point_path)
        saver = tf.train.Saver(self.var_list)
        saver.save(self.sess, check_point_path)

        self.metadata.save(path_join(path, 'meta.pkl'))
        self.metadata.save(path_join(path, 'meta.json'))

        self.params_path = path_join(path, 'params.pkl')
        self._save_params(self.params_path)

        return path

    def load(self, path):
        self.log.info(f'load from {path}')
        self.metadata.load(path_join(path, 'meta.json'))
        self._load_params(path_join(path, 'params.pkl'))

        return self

    def restore(self, path):
        self.log.info(f'restore from {path}')

        saver = tf.train.Saver(self.var_list)
        saver.restore(self.sess, path_join(path, 'check_point', 'instance.ckpt'))

    def _loss_check(self, loss_pack):
        for key, item, in loss_pack.items():
            if any(np.isnan(item)):
                self.log.error(f'{key} is nan')
                raise TrainFailError(f'{key} is nan')
            if any(np.isinf(item)):
                self.log.error(f'{key} is inf')
                raise TrainFailError(f'{key} is inf')

    def _train_iter(self, dataset, batch_size):
        raise NotImplementedError

    def train(
            self, x, y=None, epoch=1, batch_size=None,
            dataset_callback=None, epoch_pbar=True, iter_pbar=True, epoch_callbacks=None,
    ):

        if not self.is_built:
            raise RuntimeError(f'{self} not built')

        batch_size = getattr(self, 'batch_size') if batch_size is None else batch_size
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

            metric = getattr(self, 'metric', None)(x, y)
            tqdm.write(f"e:{global_epoch}, metric : {np.mean(metric)}")
            if metric in (np.nan, np.inf, -np.inf): break

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

    def train_supervised(
            self, x, y, epoch=1, batch_size=None,
            dataset_callback=None,
            epoch_pbar=True, iter_pbar=True, epoch_callbacks=None,
    ):

        if not self.is_built:
            raise RuntimeError(f'{self} not built')

        batch_size = getattr(self, 'batch_size') if batch_size is None else batch_size
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

            metric = getattr(self, 'metric', None)(x, y)
            tqdm.write(f"e:{global_epoch}, metric : {np.mean(metric)}")
            if metric in (np.nan, np.inf, -np.inf): break

            break_epoch = False
            if epoch_callbacks:
                results = [
                    callback(self, dataset, metric, global_epoch)
                    for callback in epoch_callbacks
                ]

                for result in results:
                    if result and getattr(result, 'break_epoch', False):
                        break_epoch = True
            if break_epoch: break

        if epoch_pbar: epoch_pbar.close()
        if dataset_callback: del dataset

        return metric

    def train_unsupervised(
            self, x, epoch=1, batch_size=None,
            dataset_callback=None,
            epoch_pbar=True, iter_pbar=True, epoch_callbacks=None,
    ):

        if not self.is_built:
            raise RuntimeError(f'{self} not built')

        batch_size = getattr(self, 'batch_size') if batch_size is None else batch_size
        dataset = dataset_callback if dataset_callback else BaseDataset(x=x)

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

            metric = getattr(self, 'metric', None)(x)
            tqdm.write(f"global epoch:{global_epoch}, metric : {np.mean(metric)}")
            if metric in (np.nan, np.inf, -np.inf): break

            break_epoch = False
            if epoch_callbacks:
                results = [
                    callback(self, dataset, metric, global_epoch)
                    for callback in epoch_callbacks
                ]

                for result in results:
                    if result and getattr(result, 'break_epoch', False):
                        break_epoch = True
            if break_epoch: break

        if epoch_pbar: epoch_pbar.close()
        if dataset_callback: del dataset

        return metric
