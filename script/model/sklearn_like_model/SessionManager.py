import tensorflow as tf

from script.util.MixIn import LoggerMixIn
from script.util.misc_util import error_trace


class SessionManager(LoggerMixIn):
    def __init__(self, sess=None, config=None, verbose=0):
        super().__init__(verbose)

        self.sess = sess
        self.config = config
        if sess is None:
            self.is_injected = False
        else:
            self.is_injected = True

    def __str__(self):
        return f'{self.__class__.__name__} sess={self.sess}'

    def __repr__(self):
        return f'{self.__class__.__name__} sess={self.sess}'

    def __del__(self):
        # TODO this del need hack
        self.close()

    @property
    def is_opened(self):
        return True if self.sess is not None else False

    def open(self):
        if self.sess is not None:
            raise RuntimeError(f'session already opened')

        try:
            self.sess = tf.Session(config=self.config)
        except BaseException as e:
            self.log.error(error_trace(e))

    def open_if_not(self):
        if not self.is_opened:
            self.open()

    def close(self):
        if self.sess is None:
            return

        if self.is_injected:
            raise RuntimeError(f'injected session can not close in Session manager')

        try:
            self.sess.close()
        except BaseException as e:
            self.log.error(error_trace(e))

    def init_variable(self, var_list=None):
        self.sess.run(tf.variables_initializer(var_list))
