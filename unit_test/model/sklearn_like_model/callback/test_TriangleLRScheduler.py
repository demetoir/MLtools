from script.model.sklearn_like_model.callback.TriangleLRScheduler import TriangleLRScheduler


def test_TriangleLRScheduler():
    class Model:
        def __init__(self, lr=0.01):
            self.lr = lr

        def update_learning_rate(self, x):
            self.lr = x

        @property
        def learning_rate(self):
            return self.lr

    model = Model()
    callback = TriangleLRScheduler(15, 0.01, 0.001)

    for i in range(1, 100 + 1):
        callback(model, None, None, i)
        # print(model.learning_rate)