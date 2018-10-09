import tensorflow as tf

from script.util.MixIn import LoggerMixIn


class BaseLossModule(LoggerMixIn):
    def __init__(self, name=None, verbose=0, **kwargs):
        super().__init__(verbose=verbose)
        if name is None:
            name = self.__class__.__name__
        self.name = name

    def _build(self):
        raise NotImplementedError

    @property
    def loss(self):
        raise NotImplementedError

    def build(self):
        with tf.variable_scope(self.name):
            self._build()
        self.log.info(f'build {self.name}, {self.loss}')
        return self
