from script.model.sklearn_like_model.callback.CosineResetLRScheduler import CosineResetLRScheduler


def test_CosineResetLRScheduler():
    class Model:
        def __init__(self):
            self.lr = 0.01

        def update_learning_rate(self, x):
            self.lr = x

        @property
        def learning_rate(self):
            return self.lr

    model = Model()
    callback = CosineResetLRScheduler(15, -2, -3)

    for i in range(1, 100 + 1):
        callback(model, None, None, i)
        # print(model.learning_rate)
