import tensorflow as tf

from script.util.MixIn import LoggerMixIn


class MetaBaseLossModule(type):
    """Metaclass for hook inherited class's function
    metaclass ref from 'https://code.i-harness.com/ko/q/11fc307'
    """

    def __init__(cls, name, bases, cls_dict):
        type.__init__(cls, name, bases, cls_dict)

        # hook __call__
        f_name = 'build'
        if f_name in cls_dict:
            func = cls_dict[f_name]

            def new_func(self):
                with tf.variable_scope(self.name):
                    func(self)
                self.log.info(f'build {self.name}')
                return self

            new_func.__name__ = f_name + '_wrap'
            setattr(cls, f_name, new_func)


class BaseLossModule(LoggerMixIn, metaclass=MetaBaseLossModule):
    def __init__(self, name=None, verbose=0, **kwargs):
        super().__init__(verbose=verbose)
        if name is None:
            name = self.__class__.__name__
        self.name = name

    # def _build(self):
    #     raise NotImplementedError

    @property
    def loss(self):
        raise NotImplementedError

    def build(self):
        raise NotImplementedError
        # with tf.variable_scope(self.name):
        #     self._build()
        # self.log.info(f'build {self.name}, {self.loss}')
        # return self
