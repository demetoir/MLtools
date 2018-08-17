import os
from env_settting import SKLEARN_PARAMS_SAVE_PATH
from script.sklearn_like_toolkit.param_search.HyperOpt.HyperOpt import HyperOpt_fn
from script.sklearn_like_toolkit.warpper.base.MixIn import RegWrapperMixIn, \
    MetaBaseWrapperReg
from script.util.misc_util import time_stamp, dump_pickle, load_pickle, \
    log_error_trace


# TODO remake
class regpack_HyperOpt_fn(HyperOpt_fn):

    @staticmethod
    def fn(params, feed_args, feed_kwargs):
        reg_cls = feed_kwargs['reg_cls']
        dataset = feed_kwargs['dataset']

        dataset.shuffle()
        train_set, test_set = dataset.split()
        train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
        test_Xs, test_Ys = test_set.full_batch(['Xs', 'Ys'])

        reg = reg_cls(**params)
        reg.fit(train_Xs, train_Ys)
        score = reg.score_pack(test_Xs, test_Ys)['RMSE']

        return score


class BaseWrapperRegPack(RegWrapperMixIn, metaclass=MetaBaseWrapperReg):
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

    def __getitem__(self, item):
        return self.pack.__getitem__(item)

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
                self.log.warn(f'while _collect_predict, {key} raise {e}')
        return result

    def predict(self, Xs):
        return self._collect_predict(Xs)

    def score(self, Xs, Ys, metric='accuracy'):
        Ys = self.np_arr_to_index(Ys)
        scores = {}
        for clf_k, predict in self._collect_predict(Xs).items():
            scores[clf_k] = self.pack[clf_k]._apply_metric(Ys, predict, metric)
        return scores

    def score_pack(self, Xs, Ys):
        Ys = self.np_arr_to_index(Ys)
        ret = {}
        for clf_k, predict in self._collect_predict(Xs).items():
            ret[clf_k] = self.pack[clf_k]._apply_metric_pack(Ys, predict)
        return ret

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
                log_error_trace(self.log.warn, e,
                                f'while execute confidence at {key},\n')

        return confidences

    def drop(self, key):
        self.pack.pop(key)
