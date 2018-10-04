import numpy as np

from script.model.sklearn_like_model.BaseModel import BaseEpochCallback


class EarlyStop(BaseEpochCallback):
    def __init__(self, patience, name='EarlyStop', log_func=None, min_best=True):
        self.patience = patience
        self.min_best = min_best
        self.name = name
        if log_func is None:
            log_func = print
        self.log_func = log_func

        if self.min_best:
            self.recent_best = np.Inf
        else:
            self.recent_best = -np.Inf

        self.patience_count = 0

    @property
    def patience_count_down(self):
        return self.patience - self.patience_count

    def reset_count(self):
        self.patience_count = 0

    def is_improved(self, metric):
        if self.min_best:
            result = self.recent_best > metric
        else:
            result = self.recent_best < metric
        return result

    def __call__(self, model, dataset, metric, epoch):
        if self.is_improved(metric):
            self.log_func(
                f'in {self.name}, '
                f'{getattr(self, "dc_key", "metric")} improve {abs(self.recent_best - metric)}'
            )
            self.recent_best = metric
            self.reset_count()
        else:
            self.patience_count += 1
            self.log_func(
                f'in {self.name}, '
                f'{getattr(self, "dc_key", "metric")} not improve, '
                f'patience_count_down={self.patience_count_down}'
            )

        if self.patience_count >= self.patience:
            self.reset_count()
            self.log_func(f'early stop')
            return {'break_epoch': True}
        else:
            return {}


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
