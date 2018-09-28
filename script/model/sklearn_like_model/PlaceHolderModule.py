from functools import reduce

from script.util.tensor_ops import placeholder
import tensorflow as tf


class PlaceHolderModule:
    def __init__(self, x, dtype=tf.float32, name='placeHolderModule'):
        self.dtype = dtype
        self.name = name
        self.single_shape = x.shape[1:]
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

        return self
