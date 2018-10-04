from script.model.sklearn_like_model.callback.BaseEpochCallback import BaseEpochCallback
import numpy as np


class ReduceLrOnPlateau(BaseEpochCallback):
    def __init__(self, reduce_rate, patience, min_lr=None, min_best=True, name='ReduceLrOnPlateau', log=print):
        self.reduce_rate = reduce_rate
        self.patience = patience
        self.min_lr = min_lr if min_lr else -np.inf
        self.min_best = min_best
        self.name = name
        self.log = log

        self.patience_count = 0
        self.best_metric = np.inf

    def reset_patience_count(self):
        self.patience_count = 0

    @property
    def patience_count_down(self):
        return self.patience - self.patience_count

    def __call__(self, model, dataset, metric, epoch):
        if metric < metric:
            self.reset_patience_count()
            self.best_metric = metric
            self.log(f'in {self.name}, best metric update')
        else:
            self.patience_count += 1
            self.log(f'in {self.name}, metric not improved, patience_count_down={self.patience_count_down}')

        if self.patience_count >= self.patience:
            self.reset_patience_count()
            lr = model.learning_rate
            new_lr = max(lr * self.reduce_rate, self.min_lr)
            model.update_learning_rate(new_lr)
            self.log(f'in {self.name}, reduce learning rate from {lr} to {new_lr}')
