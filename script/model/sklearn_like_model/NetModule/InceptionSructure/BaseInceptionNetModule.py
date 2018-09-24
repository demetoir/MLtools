from script.model.sklearn_like_model.NetModule.BaseNetModule import BaseNetModule


class BaseInceptionNetModule(BaseNetModule):
    def __init__(self, x, n_classes, capacity=None, reuse=False, resize_shape=(299, 299), name=None, verbose=0):
        super().__init__(capacity, reuse, name, verbose)
        self.x = x
        self.n_classes = n_classes
        self.capacity = capacity
        self.resize_shape = resize_shape

        if capacity:
            self.n_channel = capacity
        else:
            self.n_channel = 16

    def build(self):
        raise NotImplementedError
