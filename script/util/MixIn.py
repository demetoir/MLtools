from multiprocessing.pool import Pool
from script.util.Logger import Logger
from script.util.misc_util import dump_pickle, load_pickle, dump_json, load_json


class LoggerMixIn:
    def __init__(self, verbose=0):
        self.verbose = verbose

    @property
    def log(self):
        level = Logger.verbose_to_level(self.verbose)
        return Logger(self.__class__.__name__, level=level)


class PickleMixIn:
    def dump(self, path):
        dump_pickle(self, path)

    def load(self, path):
        load_obj = load_pickle(path)
        if load_obj.__class__ is not self.__class__:
            raise TypeError(f"load obj is not {load_obj.__class__} is not match with expected class {self.__class__}")
        new_obj = self.__class__()
        for key, items in load_obj.__dict__.items():
            setattr(new_obj, key, items)
        return new_obj

    def to_pickle(self, path, **kwargs):
        dump_pickle(self, path)

    def from_pickle(self, path, overwrite_self=False, **kwargs):
        load_obj = load_pickle(path)

        # for auto detect pickle type
        if load_obj.__class__ is not self.__class__:
            raise TypeError(f"load obj is not {load_obj.__class__} is not match with expected class {self.__class__}")
        new_obj = self.__class__()
        for key, items in load_obj.__dict__.items():
            setattr(new_obj, key, items)

        if overwrite_self:
            for key, val in new_obj.__dict__.items():
                self.__dict__[key] = val

        return new_obj


class JsonMixIn:
    def _dump_json(self, obj, path):
        dump_json(obj, path)

    def _load_json(self, path):
        return load_json(path)

    def _from_json(self, path):
        return load_json(path)

    def _to_json(self, obj, path):
        self._dump_json(obj, path)


class singletonPoolMixIn:
    _pool_singleton = None
    _n_job = None

    def __init__(self, n_job=1):
        self.__class__._n_job = n_job

    @property
    def pool(self):
        if self.__class__._pool_singleton is None:
            self.__class__._pool_singleton = Pool(
                processes=self.__class__._n_job)

        return self.__class__._pool_singleton
