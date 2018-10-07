from functools import reduce
from script.util.MixIn import LoggerMixIn
from script.util.tensor_ops import placeholder
import tensorflow as tf


class PlaceHolderModule(LoggerMixIn):
    def __init__(self, shape, dtype=tf.float32, name='placeHolderModule', verbose=0):
        super().__init__(verbose=verbose)
        self.dtype = dtype
        self.name = name
        self.single_shape = shape
        self.batch_shape = [None] + list(self.single_shape)
        self.flatten_size = reduce(lambda a, b: a * b, self.single_shape)

    @property
    def placeholder(self):
        return self._ph

    @property
    def shape_dict(self):
        return {
            '{self.name}_shape': self.single_shape,
            '{self.name}s_shape': self.batch_shape,
            '{self.name}_flatten_size': self.flatten_size,
        }

    def build(self):
        self._ph = placeholder(self.dtype, self.batch_shape, self.name)
        self.log.info(f'build placeholder {self.name}, {self._ph.shape}')

        return self
