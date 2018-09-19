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