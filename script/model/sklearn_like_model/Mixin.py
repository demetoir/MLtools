from functools import reduce
from tqdm import tqdm
from script.util.misc_util import load_json, dump_json, load_pickle, dump_pickle
import numpy as np
import inspect


class input_shapesMixIN:

    def __init__(self):
        if not hasattr(self, '_input_shape_keys'):
            self._input_shape_keys = []

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
        self._apply_input_shapes(load_pickle(path))

    def _save_input_shapes(self, path):
        dump_pickle(self.input_shapes, path)


class BaseInputMixIn:
    # TODO
    pass


class Xs_MixIn:
    _Xs_shapes_key = 'Xs'
    _Xs_input_shapes_keys = [
        'X_shape',
        'Xs_shape',
        'X_flatten_size',
    ]

    def __init__(self):
        if not hasattr(self, '_input_shape_keys'):
            self._input_shape_keys = []

        self._input_shape_keys += self._Xs_input_shapes_keys

        self.X_shape = None
        self.Xs_shape = None
        self.X_flatten_size = None

    @property
    def _Xs(self):
        return getattr(self, 'Xs', None)

    def _build_Xs_input_shape(self, shapes):
        shape = shapes['Xs'].shape
        X_shape = shape[1:]
        Xs_shape = [None] + list(X_shape)
        X_flatten_size = self._flatten_shape(X_shape)

        return {
            'X_shape': X_shape,
            'Xs_shape': Xs_shape,
            'X_flatten_size': X_flatten_size,
        }

    @staticmethod
    def _flatten_shape(x):
        return reduce(lambda a, b: a * b, x)


class zs_MixIn:
    _zs_shapes_key = 'zs'
    _zs_input_shape_keys = [
        'z_shape',
        'zs_shape',
    ]

    def __init__(self):
        if not hasattr(self, '_input_shape_keys'):
            self._input_shape_keys = []

        self._input_shape_keys += self._zs_input_shape_keys

        self.z_shape = None
        self.zs_shape = None

    @property
    def _zs(self):
        return getattr(self, 'zs', None)

    def _build_zs_input_shape(self, shapes):
        shape = shapes['zs']
        z_shape = shape[1:]
        zs_shape = [None] + list(z_shape)

        return {
            'z_shape': z_shape,
            'zs_shape': zs_shape
        }

    @staticmethod
    def _flatten_shape(x):
        return reduce(lambda a, b: a * b, x)

    @staticmethod
    def get_z_rand_uniform(shape):
        return np.random.uniform(-1.0, 1.0, size=shape)

    @staticmethod
    def get_z_rand_normal(shape):
        return np.random.normal(size=shape)

    def get_zs_rand_normal(self, size):
        return np.random.normal(0, .3, size=[size] + self.z_shape)

    def get_zs_rand_uniform(self, size):
        return np.random.uniform(-1.0, 1.0, size=[size] + self.z_shape)

    def get_zs_rand_beta(self, size):
        return np.random.beta(4, 4, size=[size] + self.z_shape)


class noise_MixIn:
    _noise_shapes_key = 'noise'
    _noise_input_shape_keys = [
        'noise_shape',
        'noises_shape',
    ]

    def __init__(self):
        if not hasattr(self, '_input_shape_keys'):
            self._input_shape_keys = []

        self._input_shape_keys += self._noise_input_shape_keys

        self.noise_shape = None
        self.noises_shape = None

    def _build_noise_input_shape(self, shapes):
        shape = shapes['noise']
        noise_shape = shape[1:]
        noises_shape = [None] + list(noise_shape)

        return {
            'noise_shape': noise_shape,
            'noises_shape': noises_shape
        }

    @property
    def _noises(self):
        return getattr(self, 'noises')

    def get_noises(self, shape=None, intensity=1.0):
        if shape is None:
            shape = self.noise_shape
        return np.random.normal(-1 * intensity, 1 * intensity, size=shape)


class Ys_MixIn:
    _Ys_shapes_key = 'Ys'
    _Ys_input_shapes_keys = [
        'Y_shape',
        'Ys_shape',
        'Y_flatten_size',
    ]

    def __init__(self):
        if not hasattr(self, '_input_shape_keys'):
            self._input_shape_keys = []

        self._input_shape_keys += self._Ys_input_shapes_keys

        self.Y_shape = None
        self.Ys_shape = None
        self.Y_flatten_size = None

    @property
    def _Ys(self):
        return getattr(self, 'Ys', None)

    def _build_Ys_input_shape(self, shapes):
        shape = shapes['Ys'].shape
        Y_shape = shape[1:]
        Ys_shape = [None] + list(Y_shape)
        Y_flatten_size = self._flatten_shape(Y_shape)

        return {
            'Y_shape': Y_shape,
            'Ys_shape': Ys_shape,
            'Y_flatten_size': Y_flatten_size,
        }

    @staticmethod
    def _flatten_shape(x):
        return reduce(lambda a, b: a * b, x)


class cs_MixIn:
    _cs_shapes_key = 'cs'
    _cs_input_shape_keys = [
        'c_shape',
        'cs_shape',
    ]

    def __init__(self):
        if not hasattr(self, '_input_shape_keys'):
            self._input_shape_keys = []

        self._input_shape_keys += self._cs_input_shape_keys

        self.c_shape = None
        self.cs_shape = None

    @property
    def _cs(self):
        return getattr(self, self._cs_shapes_key, None)

    def _build_cs_input_shape(self, shapes):
        shape = shapes[self._cs_shapes_key]
        c_shape = shape[1:]
        cs_shape = [None] + list(c_shape)

        return {
            'c_shape': c_shape,
            'cs_shape': cs_shape
        }

    @staticmethod
    def _flatten_shape(x):
        return reduce(lambda a, b: a * b, x)

    @staticmethod
    def get_c_rand_uniform(shape):
        return np.random.uniform(-1.0, 1.0, size=shape)

    @staticmethod
    def get_c_rand_normal(shape):
        return np.random.normal(size=shape)


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
    def __init__(self):
        if not hasattr(self, '_params_keys'):
            argspec = inspect.getfullargspec(self.__init__)
            self._params_keys = argspec.args
            self._params_keys.remove('self')

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
        self._apply_params(load_pickle(path))

    def _save_params(self, path):
        dump_pickle(self.params, path)


class loss_packMixIn:
    @staticmethod
    def format_loss_pack(pack):
        return " ".join([f"{key}={np.mean(val):.4f}" for key, val in pack.items()])

    @staticmethod
    def add_loss_pack(a, b):
        if a.keys() != b.keys():
            raise TypeError('add_loss_pack fail, a.keys() and b.keys() different')

        news_pack = {}
        for key in a:
            news_pack[key] = a[key] + b[key]

        return news_pack

    @staticmethod
    def div_loss_pack(pack, val):
        for key in pack:
            pack[key] /= val
        return pack


class supervised_trainMethodMixIn:
    def __init__(self, epoch_callback_fn=None):
        pass

    @property
    def train_ops(self):
        raise NotImplementedError


class unsupervised_trainMethodMixIn:

    def __init__(self, epoch_callback_fn=None):
        self.epoch_callback = epoch_callback_fn

    @property
    def train_ops(self):
        raise NotImplementedError


def slice_np_arr(x, size):
    return [x[i:i + size] for i in range(0, len(x), size)]


class predictMethodMixIn:
    @property
    def predict_ops(self):
        raise NotImplementedError
        # return getattr(self, 'predict_index')

    def _predict_batch(self, x):
        run_func = getattr(self, 'sess').run
        feed_dict = {
            getattr(self, '_Xs'): x
        }
        ops = self.predict_ops
        return run_func(ops, feed_dict=feed_dict)

    def predict(self, x):
        batch_size = getattr(self, 'batch_size')
        size = len(x)
        if size >= batch_size:
            xs = slice_np_arr(x, batch_size)
            tqdm.write('batch predict')
            predicts = [
                self._predict_batch(x_partial)
                for x_partial in tqdm(xs, total=len(xs))
            ]
            return np.concatenate(predicts)
        else:
            return self._predict_batch(x)


class predict_probaMethodMixIn:
    @property
    def predict_proba_ops(self):
        raise NotImplementedError

    def _predict_proba_batch(self, x):
        run_func = getattr(self, 'sess').run
        feed_dict = {
            getattr(self, '_Xs'): x
        }
        ops = self.predict_proba_ops
        return run_func(ops, feed_dict=feed_dict)

    def predict_proba(self, x):
        batch_size = getattr(self, 'batch_size')
        size = len(x)
        if size >= batch_size:
            xs = slice_np_arr(x, batch_size)
            tqdm.write('batch predict_proba')
            predicts = [
                self._predict_proba_batch(x_partial)
                for x_partial in tqdm(xs, total=len(xs))
            ]
            return np.concatenate(predicts)
        else:
            return self._predict_proba_batch(x)


class scoreMethodMixIn:

    @property
    def score_ops(self):
        raise NotImplementedError

    def _score_batch(self, x, y):
        run_func = getattr(self, 'sess').run
        feed_dict = {
            getattr(self, '_Xs'): x,
            getattr(self, '_Ys'): y
        }
        ops = self.score_ops
        return run_func(ops, feed_dict=feed_dict)

    def score(self, x, y):
        batch_size = getattr(self, 'batch_size')
        size = len(x)
        if size >= batch_size:
            xs = slice_np_arr(x, batch_size)
            ys = slice_np_arr(y, batch_size)
            tqdm.write('batch score')
            scores = np.array([np.mean(self._score_batch(x, y)) for x, y in tqdm(zip(xs, ys), total=len(xs))])
            return np.mean(scores)
        else:
            return self._score_batch(x, y)


class supervised_metricMethodMixIn:
    @property
    def metric_ops(self):
        raise NotImplementedError

    def _metric_batch(self, x, y):
        run_func = getattr(self, 'sess').run
        feed_dict = {
            getattr(self, '_Xs'): x,
            getattr(self, '_Ys'): y
        }
        ops = self.metric_ops
        return run_func(ops, feed_dict=feed_dict)

    def metric(self, x, y):
        batch_size = getattr(self, 'batch_size')
        size = len(x)
        if size >= batch_size:
            xs = slice_np_arr(x, batch_size)
            ys = slice_np_arr(y, batch_size)
            tqdm.write('batch metric')
            metrics = [np.mean(self._metric_batch(x, y)) for x, y in tqdm(zip(xs, ys), total=len(xs))]
            return np.mean(metrics)
        else:
            return np.mean(self._metric_batch(x, y))


class unsupervised_metricMethodMixIn:
    @property
    def metric_ops(self):
        raise NotImplementedError

    def _metric_batch(self, x):
        run_func = getattr(self, 'sess').run
        feed_dict = {
            getattr(self, '_Xs'): x
        }
        ops = self.metric_ops
        return run_func(ops, feed_dict=feed_dict)

    def metric(self, x):
        batch_size = getattr(self, 'batch_size')
        size = len(x)
        if size >= batch_size:
            xs = slice_np_arr(x, batch_size)
            tqdm.write('batch metric')
            metrics = [np.mean(self._metric_batch(x)) for x in tqdm(xs)]
            return np.mean(metrics)
        else:
            return self._metric_batch(x)
