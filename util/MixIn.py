from util.Logger import Logger
from util.misc_util import dump_pickle, load_pickle


class LoggerMixIn:
    @property
    def log(self):
        return Logger(self.__class__.__name__)


class PickleMixIn:
    def dump(self, path):
        dump_pickle(self, path)

    def load(self, path):
        return load_pickle(path)
