import numpy as np


class InputShape:
    def __init__(self, shape):
        # Todo check shape type..
        self.__shape = shape
        self.__batch_shape = [None] + list(np.array(self.shape))
        self.__flatten_size = int(np.prod(self.shape))

    @property
    def shape(self):
        return self.__shape

    @property
    def batch_shape(self):
        return self.__batch_shape

    @property
    def flatten_size(self):
        return self.__flatten_size

    def is_same_shape(self, a):
        return np.array_equal(np.array(a).shape, self.shape)
