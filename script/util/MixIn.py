from script.util.Logger import Logger
from script.util.misc_util import dump_pickle, load_pickle


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
