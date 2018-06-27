from script.data_handler.DummyDataset import DummyDataset
from script.util.MixIn import LoggerMixIn
from script.util.misc_util import dump_json, load_json, time_stamp, path_join, log_error_trace
from script.util.misc_util import setup_directory
from env_settting import *
from functools import reduce
import tensorflow as tf
import os
import numpy as np


class ModelBuildFailError(BaseException):
    pass


class MetaBaseModel(type):

    def __init__(cls, name, bases, cls_dict):
        type.__init__(cls, name, bases, cls_dict)

        for key in cls_dict:
            if "build_" in key:
                func = cls_dict[key]
                print(key, func)

                def wrapper(self, *args, **kwargs):
                    self.log(key)
                    return func(self, *args, **kwargs)

                setattr(cls, key, wrapper)
            # new_load = None
            # if 'load' in cls_dict:
            #     def new_load(self, path, limit):
            #         try:
            #             self.if_need_download(path)
            #             cls_dict['load'](self, path, limit)
            #             self.after_load(limit)
            #         except Exception:
            #             exc_type, exc_value, exc_traceback = sys.exc_info()
            #             err_msg = traceback.format_exception(exc_type, exc_value, exc_traceback)
            #             self.log(*err_msg)
            # setattr(cls, 'load', new_load)
        return


META_DATA_FILE_NAME = 'instance.meta'
meta_json = 'meta.json'
params_json = 'params.json'
input_shapes_json = 'input_shapes.json'
INSTANCE_FOLDER = 'instance'


class input_shapesMixIN:
    _input_shape_keys = []

    def __init__(self):
        for key in self._input_shape_keys:
            setattr(self, key, None)

    @property
    def input_shapes(self):
        return self._collect_input_shapes()

    def _apply_input_shapes(self, input_shapes):
        for key in self._input_shape_keys:
            setattr(self, key, input_shapes[key])

    def _collect_input_shapes(self):
        input_shapes = {}
        for key in self._input_shape_keys:
            input_shapes[key] = getattr(self, key, None)
        return input_shapes

    @staticmethod
    def _check_input_shapes(a, b):
        return True if dict(a) == dict(b) else False

    def _load_input_shapes(self, path):
        self._apply_input_shapes(load_json(path))

    def _save_input_shapes(self, path):
        dump_json(self.input_shapes, path)


class metadataMixIN:
    _metadata_keys = [
        'id',
        'instance_path',
        'metadata_path',
        'check_point_path',
        'save_folder_path',
    ]

    def __init__(self):
        for key in self._metadata_keys:
            setattr(self, key, None)

    @property
    def metadata(self):
        return self._collect_metadata()

    def _collect_metadata(self):
        metadata = {}
        for key in self._metadata_keys:
            metadata[key] = getattr(self, key, None)
        return metadata

    def _apply_metadata(self, metadata):
        for key in self._metadata_keys:
            setattr(self, key, metadata[key])

    def _load_metadata(self, path):
        self._apply_metadata(load_json(path))

    def _save_metadata(self, path):
        dump_json(self.metadata, path)


class paramsMixIn:
    _params_keys = []

    def __init__(self):
        for key in self._params_keys:
            setattr(self, key, None)

    @property
    def params(self):
        return self._collect_params()

    def _collect_params(self):
        params = {}
        for key in self._params_keys:
            params[key] = getattr(self, key, None)
        return params

    def _apply_params(self, params):
        for key in self._params_keys:
            setattr(self, key, params[key])

    def _load_params(self, path):
        self._apply_params(load_json(path))

    def _save_params(self, path):
        dump_json(self.params, path)


class BaseModel(LoggerMixIn, input_shapesMixIN, metadataMixIN, paramsMixIn):
    """Abstract class of model for tensorflow graph"""
    AUTHOR = 'demetoir'

    def __init__(self, verbose=10, **kwargs):
        super(BaseModel, self).__init__()
        """create instance of AbstractModel

        :param verbose:
        :type logger_path: str
        :param logger_path: path for log file
        if logger_path is None, log ony stdout
        """
        input_shapesMixIN.__init__(self)
        metadataMixIN.__init__(self)
        paramsMixIn.__init__(self)

        self.verbose = verbose
        self.sess = None
        self.saver = None
        self._is_built = False

        # gen instance id
        self.id = "_".join([self.__str__(), time_stamp()])

    def __str__(self):
        return "%s_%s" % (self.AUTHOR, self.__class__.__name__)

    def __del__(self):
        # TODO this del need hack
        try:
            self._close_session()
            self._close_saver()
            # reset tensorflow graph
            tf.reset_default_graph()

        except BaseException:
            pass

    def _open_saver(self):
        if self.saver is None:
            self.saver = tf.train.Saver()

    def _close_saver(self):
        self.saver = None

    def _open_session(self):
        try:
            if self.sess is None:
                self.sess = tf.Session()
                self.sess.run(tf.global_variables_initializer())
        except BaseException as e:
            log_error_trace(self.log.error, e, head='fail to open tf.Session()')

    def _close_session(self):
        if self.sess is not None:
            self.sess.close()
            self.sess = None

    def _build(self):
        try:
            with tf.variable_scope(str(self.id)):
                with tf.variable_scope("misc_ops"):
                    self.log.debug("build_misc_ops")
                    self._build_misc_ops()

                self.log.debug('build_main_graph')
                self._build_main_graph()

                with tf.variable_scope('loss_function'):
                    self.log.debug('build_loss_function')
                    self._build_loss_function()

                with tf.variable_scope('train_ops'):
                    self.log.debug('build_train_ops')
                    self._build_train_ops()

        except BaseException as e:
            log_error_trace(self.log.error, e)
            log_error_trace(print, e)
            raise ModelBuildFailError("ModelBuildFailError")
        else:
            self._is_built = True
            self.log.info("build success")

    def _build_input_shapes(self, input_shapes):
        """load input shapes for tensor placeholder

        :type input_shapes: dict
        :param input_shapes: input shapes for tensor placeholder

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
        self.global_step = tf.get_variable("global_step", shape=1, initializer=tf.zeros_initializer)
        self.op_inc_global_step = tf.assign(self.global_step, self.global_step + 1, name='op_inc_global_step')

        self.global_epoch = tf.get_variable("global_epoch", shape=1, initializer=tf.zeros_initializer)
        self.op_inc_global_step = tf.assign(self.global_epoch, self.global_step + 1, name='op_inc_global_epoch')

    def _build_train_ops(self):
        """Load train operation of model

        :raise NotImplementError
        if not implemented
        """
        raise NotImplementedError

    def _prepare_train(self, **kwargs):
        shapes = {}
        for key in kwargs:
            shapes[key] = kwargs[key].shape[1:]
        input_shapes = self._build_input_shapes(shapes)
        self._apply_input_shapes(input_shapes)
        self.is_built()

    def save(self, path=None):
        if path is None:
            self.log.info('save directory not specified, use default directory')
            path = os.path.join(ROOT_PATH, 'instance', self.id)

        self.instance_path = path
        self.save_folder_path = path_join(self.instance_path, 'check_point')
        self.check_point_path = path_join(self.save_folder_path, 'instance.ckpt')
        self.metadata_path = path_join(self.instance_path, 'meta.json')
        setup_directory(self.instance_path)
        setup_directory(self.save_folder_path)
        self._save_metadata(self.metadata_path)

        self.input_shapes_path = path_join(self.instance_path, 'input_shapes.json')
        self._save_input_shapes(self.input_shapes_path)
        self.params_path = path_join(self.instance_path, 'params.json')
        self._save_params(self.params_path)

        self._open_session()
        self._open_saver()

        self.saver.save(self.sess, self.check_point_path)

        self.log.info("saved at {}".format(self.instance_path))

        return self.instance_path

    def load(self, path):
        self._load_metadata(os.path.join(path, 'meta.json'))
        self._load_input_shapes(os.path.join(path, 'input_shapes.json'))
        self._load_params(os.path.join(path, 'params.json'))

        self._build()

        self._close_session()
        self._open_session()

        self._close_saver()
        self._open_saver()
        self.saver.restore(self.sess, self.check_point_path)

    def get_tf_values(self, fetches, feet_dict):
        self.is_built()

        return self.sess.run(fetches, feet_dict)

    def is_built(self):
        if not self._is_built:
            self._build()

        if self.sess is None:
            self._open_session()

    @staticmethod
    def to_dummyDataset(**kwargs):
        dataset = DummyDataset()
        for key, item in kwargs.items():
            dataset.add_data(key, item)
        return dataset

    @staticmethod
    def shape_extract(**kwargs):
        ret = {}
        for key, item in kwargs.items():
            shape = np.array(item).shape
            if len(shape) == 1:
                ret[key] = shape
            else:
                ret[key] = shape[1:]
        return ret

    @staticmethod
    def flatten_shape(x):
        return reduce(lambda a, b: a * b, x)
