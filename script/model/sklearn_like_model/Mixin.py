from functools import reduce
from script.util.misc_util import load_json, dump_json
import numpy as np


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
    def get_z_rand(shape):
        return np.random.uniform(-1.0, 1.0, size=shape)


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
    _Xs_shapes_key = 'Ys'
    _Xs_input_shapes_keys = [
        'Y_shape',
        'Ys_shape',
        'Y_flatten_size',
    ]

    def __init__(self):
        if not hasattr(self, '_input_shape_keys'):
            self._input_shape_keys = []

        self._input_shape_keys += self._Xs_input_shapes_keys

        self.Y_shape = None
        self.Ys_shape = None
        self.Y_flatten_size = None

    @property
    def _Ys(self):
        return getattr(self, 'Ys', None)

    def _build_Xs_input_shape(self, shapes):
        shape = shapes['Ys'].shape
        X_shape = shape[1:]
        Xs_shape = [None] + list(X_shape)
        X_flatten_size = self._flatten_shape(X_shape)

        return {
            'Y_shape': X_shape,
            'Ys_shape': Xs_shape,
            'Y_flatten_size': X_flatten_size,
        }

    @staticmethod
    def _flatten_shape(x):
        return reduce(lambda a, b: a * b, x)


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
