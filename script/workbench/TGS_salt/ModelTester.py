from script.util.misc_util import error_trace


class ModelTester:
    def __init__(self, model, inputs, path=None, **kwargs):
        self.model = model
        self.inputs = inputs
        self.path = path
        self.kwargs = kwargs

    def build_test(self):
        try:
            self.model.build(**self.inputs)
        except BaseException as e:
            print(error_trace(e))
            raise RuntimeError(f'build test fail\n{self.model}')

    def train_test(self):
        raise NotImplementedError

    def load_restore_test(self):
        try:
            if self.path is None:
                path = './test_instance'
            else:
                path = self.path

            self.model.save(path)
            self.model.load(path)
            self.model.restore(path)
        except BaseException as e:
            print(error_trace(e))
            raise RuntimeError(f'load_restore_test fail\n{self.model}')
