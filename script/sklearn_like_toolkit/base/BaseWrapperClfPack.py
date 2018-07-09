import os
from pprint import pformat
from env_settting import SKLEARN_PARAMS_SAVE_PATH
from script.data_handler.DummyDataset import DummyDataset
from script.sklearn_like_toolkit.HyperOpt.HyperOpt import HyperOpt, HyperOpt_fn
from script.sklearn_like_toolkit.ParamOptimizer import ParamOptimizer
from script.sklearn_like_toolkit.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.base.MixIn import ClfWrapperMixIn, meta_BaseWrapperClf
from script.sklearn_like_toolkit.warpper.wrapperGridSearchCV import wrapperGridSearchCV
from script.util.misc_util import time_stamp, dump_pickle, load_pickle, path_join, log_error_trace


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

    def __init__(self, pack_keys=None):
        super().__init__()

        self.pack = {}
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

    def HyperOptSearch(self, Xs, Ys, n_iter, min_best=False, parallel=False, **kwargs):
        Ys = self.np_arr_to_index(Ys)

        dataset = DummyDataset()
        dataset.add_data('Xs', Xs)
        dataset.add_data('Ys', Ys)

        total = len(self.pack)
        current = 0
        for key, clf in self.pack.items():
            current += 1
            try:
                self.log.info(f'HyperOpt at {key} {current}/{total}')
                opt = HyperOpt()

                if parallel:
                    trials = opt.fit_parallel(
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

                else:
                    trials = opt.fit_serial(
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
                    clf.fit(Xs, Ys)
                    self.pack[key] = clf

            except BaseException as e:
                log_error_trace(self.log.warn, e, head=f'while HyperOpt at {key}')
                self.log.warn(f'while, HyperOpt at {key}, raise ')

    def param_search(self, Xs, Ys):
        result_csv_path = path_join('.', 'param_search_result', time_stamp())
        Ys = self.np_arr_to_index(Ys)
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

    def gridSearchCV(self, Xs, Ys, **kwargs):
        Ys = self.np_arr_to_index(Ys)

        total = len(self.pack)
        current = 0
        for key, clf in self.pack.items():
            current += 1
            try:
                self.log.info(f'gridSearchCV at {key} {current}/{total}')
                optimizer = wrapperGridSearchCV(clf, clf.tuning_grid, **kwargs)
                optimizer.fit(Xs, Ys)
                self.pack[key] = optimizer.best_estimator_
                self.optimize_result = optimizer.cv_results_
                # self.optimizers[key] = optimizer
            except BaseException as e:
                log_error_trace(self.log.warn, e, head=f'while GridSearchCV at {key}')
                self.log.warn(f'while, GridSearchCV at {key}, raise ')

    def fit(self, Xs, Ys):
        Ys = self.np_arr_to_index(Ys)
        for key in self.pack:
            try:
                self.pack[key].fit(Xs, Ys)
            except BaseException as e:
                log_error_trace(self.log.warn, e)
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
        Ys = self.np_arr_to_index(Ys)
        scores = {}
        for clf_k, predict in self._collect_predict(Xs).items():
            scores[clf_k] = self._apply_metric(Ys, predict, metric)
        return scores

    def score_pack(self, Xs, Ys):
        Ys = self.np_arr_to_index(Ys)
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
        for key, clf in self.pack.items():
            try:
                confidences[key] = clf.predict_confidence(Xs)
            except BaseException as e:
                log_error_trace(self.log.warn, e, f'while execute confidence at {key},\n')

        return confidences

    def drop(self, key):
        self.pack.pop(key)
