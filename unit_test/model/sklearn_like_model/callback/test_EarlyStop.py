import numpy as np


def train_early_stop(self, x, y, n_epoch=200, patience=20, min_best=True):
    if min_best is False:
        raise NotImplementedError

    last_metric = np.Inf
    patience_count = 0
    for e in range(1, n_epoch + 1):
        self.train(x, y, epoch=1)
        metric = self.metric(x, y)
        print(f'e = {e}, metric = {metric}, best = {last_metric}')

        if last_metric > metric:
            print(f'improve {last_metric - metric}')
            last_metric = metric
            patience_count = 0
        else:
            patience_count += 1

        if patience_count == patience:
            print(f'early stop')
            break


def test_train_early_stop():
    class dummy_model:
        def train(self, x, y, epoch=4):
            pass

        def metric(self, x, y):
            import random
            return random.uniform(0, 10)

    x = None
    y = None
    model = dummy_model()
    train_early_stop(model, x, y, n_epoch=100)
