from pprint import pformat
from tqdm import tqdm
from env_settting import SKLEARN_PARAMS_SAVE_PATH
from script.data_handler.DummyDataset import DummyDataset
from script.sklearn_like_toolkit.HyperOpt.HyperOpt import HyperOpt, HyperOpt_fn
from script.sklearn_like_toolkit.ParamOptimizer import ParamOptimizer
from script.sklearn_like_toolkit.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.base.MixIn import ClfWrapperMixIn, meta_BaseWrapperClf
from script.sklearn_like_toolkit.warpper.wrapperGridSearchCV import wrapperGridSearchCV
from script.util.misc_util import time_stamp, path_join, log_error_trace
import numpy as np


class clfpack_HyperOpt_fn(HyperOpt_fn):
    @staticmethod
    def fn(params, feed_args, feed_kwargs):
        clf_cls = feed_kwargs['clf_cls']
        dataset = feed_kwargs['dataset']

        dataset.shuffle()
        train_set, test_set = dataset.split()
        train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
        test_Xs, test_Ys = test_set.full_batch(['Xs', 'Ys'])

        clf = clf_cls(**params)
        clf.fit(train_Xs, train_Ys)
        score = clf.score(test_Xs, test_Ys)

        return score


class BaseWrapperClfPack(ClfWrapperMixIn, metaclass=meta_BaseWrapperClf):
    class_pack = {}

    def __init__(self, pack_keys=None, n_classes=None):
        super().__init__()

        self.pack = {}
        self.n_classes = n_classes

        self.optimizers = {}
        self.optimize_result = {}
        self.params_save_path = SKLEARN_PARAMS_SAVE_PATH

        self._HyperOpt_trials = {}
        self._HyperOpt_losses = {}
        self._HyperOpt_results = {}

        self._HyperOpt_best_result = {}
        self._HyperOpt_best_loss = {}
        self._HyperOpt_best_params = {}

        if pack_keys is None:
            pack_keys = self.class_pack.keys()

        self.pack = {}
        for key in pack_keys:
            self.pack[key] = self.class_pack[key]()

    def __str__(self):
        return self.__class__.__name__

    def __getitem__(self, item) -> BaseWrapperClf:
        return self.pack.__getitem__(item)

    @property
    def HyperOpt_results(self):
        return self._HyperOpt_results

    @property
    def HyperOpt_Trials(self):
        return self._HyperOpt_trials

    @property
    def HyperOpt_losses(self):
        return self._HyperOpt_losses

    @property
    def HyperOpt_best_loss(self):
        return self._HyperOpt_best_loss

    @property
    def HyperOpt_best_params(self):
        return self._HyperOpt_best_params

    @property
    def HyperOpt_best_result(self):
        return self._HyperOpt_best_result

    @property
    def HyperOpt_opt_info(self):
        return {key: self.optimizers[key].opt_info for key in self.pack}

    def HyperOptSearch(self, x, y, n_iter, min_best=False, parallel=False, **kwargs):
        y = self.np_arr_to_index(y)

        dataset = DummyDataset()
        dataset.add_data('Xs', x)
        dataset.add_data('Ys', y)

        total = len(self.pack)
        current = 0
        for key, clf in self.pack.items():
            current += 1
            try:
                tqdm.write(f'HyperOpt at {key} {current}/{total}')
                # self.log.info(f'HyperOpt at {key} {current}/{total}')
                opt = HyperOpt()

                if parallel:
                    opt_func = opt.fit_parallel
                else:
                    opt_func = opt.fit_serial

                trials = opt_func(
                    clfpack_HyperOpt_fn,
                    clf.HyperOpt_space,
                    n_iter,
                    feed_args=(),
                    feed_kwargs={
                        'clf_cls': clf.__class__,
                        'dataset': dataset
                    },
                    min_best=min_best
                )
                self.optimizers[key] = opt
                self._HyperOpt_trials[key] = trials
                self._HyperOpt_results[key] = trials.results
                self._HyperOpt_losses[key] = trials.losses
                self._HyperOpt_best_params[key] = opt.best_param
                self._HyperOpt_best_loss[key] = opt.best_loss
                self._HyperOpt_best_result[key] = opt.best_result
                self.optimize_result[key] = opt.result

                if opt.best_param is not None:
                    clf = clf.__class__(**opt.best_param)
                    clf.fit(x, y)
                    self.pack[key] = clf

            except BaseException as e:
                log_error_trace(self.log.warn, e, head=f'while HyperOpt at {key}')
                self.log.warn(f'while, HyperOpt at {key}, raise ')
        y = self.np_arr_to_index(y)

        dataset = DummyDataset()
        dataset.add_data('Xs', x)
        dataset.add_data('Ys', y)

        total = len(self.pack)
        current = 0
        for key, clf in self.pack.items():
            current += 1
            try:
                tqdm.write(f'HyperOpt at {key} {current}/{total}')
                # self.log.info(f'HyperOpt at {key} {current}/{total}')
                opt = HyperOpt()

                if parallel:
                    opt_func = opt.fit_parallel
                else:
                    opt_func = opt.fit_serial

                trials = opt_func(
                    clfpack_HyperOpt_fn,
                    clf.HyperOpt_space,
                    n_iter,
                    feed_args=(),
                    feed_kwargs={
                        'clf_cls': clf.__class__,
                        'dataset': dataset
                    },
                    min_best=min_best
                )
                self.optimizers[key] = opt
                self._HyperOpt_trials[key] = trials
                self._HyperOpt_results[key] = trials.results
                self._HyperOpt_losses[key] = trials.losses
                self._HyperOpt_best_params[key] = opt.best_param
                self._HyperOpt_best_loss[key] = opt.best_loss
                self._HyperOpt_best_result[key] = opt.best_result
                self.optimize_result[key] = opt.result

                if opt.best_param is not None:
                    clf = clf.__class__(**opt.best_param)
                    clf.fit(x, y)
                    self.pack[key] = clf

            except BaseException as e:
                log_error_trace(self.log.warn, e, head=f'while HyperOpt at {key}')
                self.log.warn(f'while, HyperOpt at {key}, raise ')

    def param_search(self, x, y):
        result_csv_path = path_join('.', 'param_search_result', time_stamp())
        y = self.np_arr_to_index(y)
        for key in self.pack:
            cls = self.pack[key].__class__
            obj = cls()

            optimizer = ParamOptimizer(obj, obj.tuning_grid)
            self.pack[key] = optimizer.optimize(x, y)
            self.optimize_result[key] = optimizer.result

            path = path_join(result_csv_path, cls.__name__ + '.csv')
            optimizer.result_to_csv(path)

            self.log.info("top 5 result")
            for result in optimizer.top_k_result():
                self.log.info(pformat(result))

    def gridSearchCV(self, x, y, **kwargs):
        y = self.np_arr_to_index(y)

        total = len(self.pack)
        current = 0
        for key, clf in self.pack.items():
            current += 1
            try:
                self.log.info(f'gridSearchCV at {key} {current}/{total}')
                optimizer = wrapperGridSearchCV(clf, clf.tuning_grid, **kwargs)
                optimizer.fit(x, y)
                self.pack[key] = optimizer.best_estimator_
                self.optimize_result = optimizer.cv_results_
                # self.optimizers[key] = optimizer
            except BaseException as e:
                log_error_trace(self.log.warn, e,
                                head=f'while GridSearchCV at {key}')
                self.log.warn(f'while, GridSearchCV at {key}, raise ')

    def _check_n_class(self, y):
        onehot_Ys = self.np_arr_to_onehot(y)
        if self.n_classes is None:
            if onehot_Ys.ndim == 1:
                self.n_classes = onehot_Ys.shape[0]
            elif onehot_Ys.ndim == 2:
                self.n_classes = onehot_Ys.shape[1]
            else:
                raise TypeError(f'y expect shape (-1) or (-1,-1), but {onehot_Ys.shape}')

        label_Ys = self.np_arr_to_index(y)
        print(np.unique(label_Ys, axis=0).shape[0], onehot_Ys.shape[1])

        feed_n_classes = np.unique(label_Ys, axis=0).shape[0]
        if feed_n_classes != self.n_classes:
            raise TypeError(f'expect {self.n_classes} class, but only {feed_n_classes} class')

    def fit(self, x, y):
        self._check_n_class(y)
        label_Ys = self.np_arr_to_index(y)

        for key in tqdm(self.pack):
            tqdm.write(f'fit {key}')

            try:
                self.pack[key].fit(x, label_Ys)
            except BaseException as e:
                log_error_trace(self.log.warn, e)
                self.log.warn(f'while fitting, {key} raise {e}')

    def predict(self, x):
        result = {}
        for key in tqdm(self.pack):
            tqdm.write(f'predict {key}')

            try:
                result[key] = self.pack[key].predict(x)
            except BaseException as e:
                self.log.warn(f'while fitting, {key} raise {e}')
        return result

    def predict_proba(self, x):
        result = {}
        for key in tqdm(self.pack):
            tqdm.write(f'predict_proba {key}')

            try:
                result[key] = self.pack[key].predict_proba(x)
            except BaseException as e:
                self.log.warn(f'while predict_proba, {key} raise {e}')
        return result

    def predict_confidence(self, x):
        confidences = {}
        for key, clf in tqdm(self.pack.items()):
            tqdm.write(f'predict_confidence {key}')

            try:
                confidences[key] = clf.predict_confidence(x)
            except BaseException as e:
                log_error_trace(self.log.warn, e,
                                f'while execute confidence at {key},\n')

        return confidences

    def score(self, x, y, metric='accuracy'):
        y = self.np_arr_to_index(y)

        scores = {}
        for clf_k, predict in tqdm(self.predict(x).items()):
            tqdm.write(f'score {clf_k}')

            scores[clf_k] = self._apply_metric(y, predict, metric)
        return scores

    def score_pack(self, x, y):
        y = self.np_arr_to_index(y)

        ret = {}
        for clf_k, predict in tqdm(self.predict(x).items()):
            tqdm.write(f'score_pack {clf_k}')

            ret[clf_k] = self._apply_metric_pack(y, predict)
        return ret

    @property
    def feature_importances(self):
        ret = {}
        for key, clf in tqdm(self.pack.items()):
            tqdm.write(f'collect feature importances {key}')

            if hasattr(clf, 'feature_importances_'):
                ret[key] = getattr(clf, 'feature_importances_')
        return ret

    def save(self, path):
        self.log.info(f'pickle save at {path}')
        super().dump(path)

    def dump(self, path):
        self.log.info(f'pickle save at {path}')
        super().dump(path)

    def drop(self, key):
        self.pack.pop(key)
