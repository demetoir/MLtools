import numpy as np
from script.model.sklearn_like_model.callback.BaseEpochCallback import BaseEpochCallback
from script.util.misc_util import path_join, setup_directory


class BestSave(BaseEpochCallback):
    def __init__(self, path, max_best=True, name='top_k_save', log=print):
        self.path = path
        self.max_best = max_best
        self.name = name
        self.log = log

        self.top_k_json_path = path_join(self.path, 'top_k.json')
        if self.max_best:
            self.best_metric = np.Inf
        else:
            self.best_metric = -np.Inf

        setup_directory(self.path)

    def __call__(self, model, dataset, metric, epoch):
        if self.max_best:
            res = metric > self.best_metric
        else:
            res = metric < self.best_metric

        if res:
            self.log(f'metric improve {self.best_metric} to {metric}({abs(self.best_metric - metric)})')
            self.best_metric = metric
            model.save(self.path)
        else:
            self.log(f'metric not improve {self.best_metric}')
