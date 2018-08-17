import numpy as np
from env_settting import SKLEARN_PARAMS_SAVE_PATH
from script.data_handler.Base.BaseDataset import BaseDataset
from script.sklearn_like_toolkit.param_search.HyperOpt.HyperOpt import HyperOpt, HyperOpt_fn
from script.sklearn_like_toolkit.warpper.base.MixIn import DFEncoderMixIn, YLabelOneHotConvertMixIn
from script.util.MixIn import LoggerMixIn


class SupervisedHyperOptCVFunc(HyperOpt_fn):
    @staticmethod
    def fn(params, feed_args, feed_kwargs):
        clf_cls = feed_kwargs['clf_cls']
        data_set = feed_kwargs['dataset']
        cv = feed_kwargs['cv']
        # TODO add metric func
        # metric = feed_kwargs['metric']
        split_rate = feed_kwargs['split_rate']

        scores = []
        for i in range(cv):
            data_set.shuffle()
            train_set, test_set = data_set.split(ratio=split_rate)
            train_x, train_y = train_set.full_batch()
            test_x, test_y = test_set.full_batch()

            clf = clf_cls(**params)
            clf.fit(train_x, train_y)
            score = clf.score(test_x, test_y)
            scores += [score]

        scores = np.array(scores)
        mean_score = np.mean(scores)

        return {'loss': mean_score, 'losses': scores}


class ParamOpt(DFEncoderMixIn, YLabelOneHotConvertMixIn, LoggerMixIn):

    def __init__(self, func=None, cv=3, n_iter=None, opt_method='HyperOpt', min_best=True,
                 split_rate=(7, 3), n_jobs=1, x_df_encoder=None, y_df_encoder=None, clone=True, verbose=0):
        DFEncoderMixIn.__init__(self, x_df_encoder, y_df_encoder)
        YLabelOneHotConvertMixIn.__init__(self)
        LoggerMixIn.__init__(self, verbose)

        if func:
            self.func = func
        else:
            self.func = SupervisedHyperOptCVFunc.fn
        self.cv = cv
        self.opt_method = opt_method
        self.opt_funcs = {
            'HyperOpt': self.HyperOptSearchCV,
        }
        self.n_iter = n_iter
        self.min_best = min_best
        self.n_jobs = n_jobs
        self.clone = clone
        self.split_rate = split_rate

        self.optimizers = {}
        self.optimize_result = {}
        self.params_save_path = SKLEARN_PARAMS_SAVE_PATH

    @staticmethod
    def _refit(clf, x, y, **kwargs):
        dataset = BaseDataset(x=x, y=y)
        dataset.shuffle()
        train_x, train_y = dataset.full_batch()

        clf.fit(train_x, train_y)
        return clf

    def HyperOptSearchCV(self, clf, x, y):
        n_iter = self.n_iter
        min_best = self.min_best
        if self.n_jobs > 1:
            parallel = True
        else:
            parallel = False

        x, y = self._if_df_encode(x, y)
        y = self.np_arr_to_index(y)
        data_set = BaseDataset(x=x, y=y)

        opt = HyperOpt(min_best, n_jobs=self.n_jobs)

        if parallel:
            opt_func = opt.fit_parallel
        else:
            opt_func = opt.fit_serial

        trials = opt_func(
            SupervisedHyperOptCVFunc,
            clf.HyperOpt_space,
            n_iter,
            feed_args=(),
            feed_kwargs={
                'clf_cls': clf.__class__,
                'dataset': data_set,
                'cv': self.cv,
                'metric': None,
                'split_rate': self.split_rate
            },
            min_best=min_best
        )
        self.optimize_result = trials
        self.optimizers = opt
        clf_optimized = clf.__class__(**opt.best_param)

        return self._refit(clf_optimized, x, y)

    def RandomSearchCv(self, clf, x, y, **kwargs):
        # TODO
        raise NotImplementedError

    def _fit(self, clf, x, y):
        x, y = self._if_df_encode(x, y)
        y = self.np_arr_to_index(y)

        opt_func = self.opt_funcs[self.opt_method]
        clf_optimized = opt_func(clf, x, y)

        return clf_optimized

    def fit(self, clf, x, y):
        return self._fit(clf, x, y)
