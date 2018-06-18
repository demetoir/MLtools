import logging
import logging.handlers
import os
from util.misc_util import time_stamp, check_path

CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
NOTSET = 0

_levelToName = {
    CRITICAL: 'CRITICAL',
    ERROR: 'ERROR',
    WARNING: 'WARNING',
    INFO: 'INFO',
    DEBUG: 'DEBUG',
    NOTSET: 'NOTSET',
}
_nameToLevel = {
    'CRITICAL': CRITICAL,
    'FATAL': FATAL,
    'ERROR': ERROR,
    'WARN': WARNING,
    'WARNING': WARNING,
    'INFO': INFO,
    'DEBUG': DEBUG,
    'NOTSET': NOTSET,
}


# catch *args and make to str
def deco_args_to_str(func):
    def wrapper(*args, **kwargs):
        return func(" ".join(map(str, args)), **kwargs)

    wrapper.__name__ = func.__name__
    return wrapper


class Logger:
    """wrapper class of logging module

    # usage
    logger = Logger("i am logger")
    log = logger.get_logger()
    log("logging message")
    log("also", "can use like print",",the build-in function",)
    """
    FILE_LOGGER_FORMAT = '%(asctime)s [%(levelname)s]> %(message)s'
    STDOUT_LOGGER_FORMAT = '%(asctime)s> %(message)s'
    EMPTY_FORMAT = ""

    def __init__(self, name, path=None, file_name=None, level='INFO', with_file=True, empty_stdout_format=True,
                 print_start_msg=True):
        """create logger

        :param name:name of logger
        :param path: log file path
        :param file_name: log file name
        :param level: logging level
        :param with_file: default False
        if std_only is True, log message print only stdout not in logfile
        """
        self.name = name
        self.logger = logging.getLogger(name + time_stamp())
        self.logger.setLevel('DEBUG')

        self.file_handler = None
        if with_file:
            if path is None:
                path = os.path.join('.', 'log')
            if file_name is None:
                file_name = "{name}.log".format(name=name)

            check_path(path)
            self.file_handler = logging.FileHandler(os.path.join(path, file_name))
            self.file_handler.setFormatter(logging.Formatter(self.FILE_LOGGER_FORMAT))
            self.file_handler.setLevel('DEBUG')
            self.logger.addHandler(self.file_handler)

        if empty_stdout_format:
            format_ = self.EMPTY_FORMAT
        else:
            format_ = self.STDOUT_LOGGER_FORMAT
        formatter = logging.Formatter(format_)
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setFormatter(formatter)
        self.stream_handler.setLevel(level)
        self.logger.addHandler(self.stream_handler)

        self._fatal = deco_args_to_str(getattr(self.logger, 'fatal'))
        self._error = deco_args_to_str(getattr(self.logger, 'error'))
        self._warn = deco_args_to_str(getattr(self.logger, 'warn'))
        self._info = deco_args_to_str(getattr(self.logger, 'info'))
        self._debug = deco_args_to_str(getattr(self.logger, 'debug'))
        self._critical = deco_args_to_str(getattr(self.logger, 'critical'))

        if print_start_msg:
            self.logger.debug('')
            self.logger.debug(f'#' * 80)
            self.logger.debug(f'start logger {name}')
            self.logger.debug(f'#' * 80)

    def __repr__(self):
        return self.__class__.__name__

    def __del__(self):
        # TODO this del need hack
        try:
            if self.file_handler is not None:
                self.logger.removeHandler(self.file_handler)
            if self.stream_handler is not None:
                self.logger.removeHandler(self.stream_handler)

            del self.logger
        except BaseException:
            pass

    def get_log(self, level='info'):
        """return logging function

        :param level: level of logging
        :return: log function
        """
        func = deco_args_to_str(getattr(self.logger, level))
        return func

    def fatal(self, *args, **kwargs):
        self._fatal(*args, **kwargs)

    def error(self, *args, **kwargs):
        self._error(*args, **kwargs)

    def warn(self, *args, **kwargs):
        self._warn(*args, **kwargs)

    def info(self, *args, **kwargs):
        self._info(*args, **kwargs)

    def debug(self, *args, **kwargs):
        self._debug(*args, **kwargs)

    def critical(self, *args, **kwargs):
        self._critical(*args, **kwargs)


class StdoutOnlyLogger(Logger):
    def __init__(self, name=None, level='DEBUG'):
        if name is None:
            name = self.__class__.__name__

        super().__init__(name, with_file=False, empty_stdout_format=True, level=level)


def pprint_logger(log_func):
    def wrapper(*args, **kwargs):
        import pprint
        return log_func(pprint.pformat(args, **kwargs))

    return wrapper


"""
self.logger = Logger(self.__class__.__name__)
self.log = self.logger.get_log()

self.logger = StdoutOnlyLogger(self.__class__.__name__)
self.log = self.logger.get_log()
"""
