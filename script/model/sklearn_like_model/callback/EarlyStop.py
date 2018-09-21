import numpy as np


class EarlyStop:
    def __init__(self, patience, log_func=print):
        self.patience = patience
        self.recent_best = np.Inf
        self.patience_count = 0
        self.log_func = log_func

    def __call__(self, metric, epoch):
        self.log_func(f'e = {epoch}, metric = {metric}, recent best = {self.recent_best}')

        if self.recent_best > metric:
            self.log_func(f'improve {self.recent_best - metric}')
            self.recent_best = metric
            self.patience_count = 0
        else:
            self.patience_count += 1

        if self.patience_count == self.patience:
            self.log_func(f'early stop')
            return True
        else:
            return False


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