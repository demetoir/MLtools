import os
from pprint import pformat
from env_settting import SKLEARN_PARAMS_SAVE_PATH
from sklearn_like_toolkit.ParamOptimizer import ParamOptimizer
from sklearn_like_toolkit.base.MixIn import Reformat_Ys_MixIn, clf_metric_MixIn
from util.MixIn import LoggerMixIn, PickleMixIn
from util.misc_util import time_stamp, dump_pickle, load_pickle, path_join
import numpy as np


class BaseWrapperPack(Reformat_Ys_MixIn, clf_metric_MixIn, LoggerMixIn, PickleMixIn):
    class_pack = {}

    def __init__(self):
        super().__init__()
        self.pack = {}
        self.optimize_result = {}
        self.params_save_path = SKLEARN_PARAMS_SAVE_PATH

    def __str__(self):
        return self.__class__.__name__

    def param_search(self, Xs, Ys):
        result_csv_path = path_join('.', 'param_search_result', time_stamp())
        Ys = self._reformat_to_index(Ys)
        for key in self.pack:
            cls = self.pack[key].__class__
            obj = cls()

            optimizer = ParamOptimizer(obj, obj.tuning_grid)
            self.pack[key] = optimizer.optimize(Xs, Ys)
            self.optimize_result[key] = optimizer.result

            path = path_join(result_csv_path, cls.__name__ + '.csv')
            optimizer.result_to_csv(path)

            self.log.info("top 5 result")
            for result in optimizer.top_k_result():
                self.log.info(pformat(result))

    def fit(self, Xs, Ys):
        Ys = self._reformat_to_index(Ys)
        for key in self.pack:
            try:
                self.pack[key].fit(Xs, Ys)
            except BaseException as e:
                self.log.warn(f'while fitting, {key} raise {e}')

    def _collect_predict(self, Xs):
        result = {}
        for key in self.pack:
            try:
                result[key] = self.pack[key].predict(Xs)
            except BaseException as e:
                self.log.warn(f'while fitting, {key} raise {e}')
        return result

    def predict(self, Xs):
        return self._collect_predict(Xs)

    def score(self, Xs, Ys, metric='accuracy'):
        Ys = self._reformat_to_index(Ys)
        scores = {}
        for clf_k, predict in self._collect_predict(Xs).items():
            scores[clf_k] = self._apply_metric(Ys, predict, metric)
        return scores

    def score_pack(self, Xs, Ys):
        Ys = self._reformat_to_index(Ys)
        ret = {}
        for clf_k, predict in self._collect_predict(Xs).items():
            ret[clf_k] = self._apply_metric_pack(Ys, predict)
        return ret

    def predict_proba(self, Xs):
        result = {}
        for key in self.pack:
            try:
                result[key] = self.pack[key].predict_proba(Xs)
            except BaseException as e:
                self.log.warn(f'while predict_proba, {key} raise {e}')
        return result

    def import_params(self, params_pack):
        for key in self.pack:
            class_ = self.class_pack[key]
            self.pack[key] = class_(**params_pack[key])

    def export_params(self):
        params = {}
        for key in self.pack:
            clf = self.pack[key]
            params[key] = clf.get_params()
        return params

    def save_params(self, path=None):
        if path is None:
            path = os.path.join(self.params_save_path, time_stamp())

        params = self.export_params()

        pickle_path = path + '.pkl'
        dump_pickle(params, pickle_path)

        self.log.info('save params at {}'.format([pickle_path]))

        return pickle_path

    def load_params(self, path):
        self.log.info('load params from {}'.format(path))
        params = load_pickle(path)

        self.import_params(params)

    def dump(self, path):
        self.log.info(f'pickle save at {path}')
        super().dump(path)

    def predict_confidence(self, Xs):
        confidences = {}
        for clf_k, proba in self.predict_proba(Xs).items():
            n_class = proba.shape[1]
            np_arr = np.abs(1.0 / n_class - proba)
            np_arr = np_arr.sum(axis=1)
            confidences[clf_k] = np_arr

        return confidences
