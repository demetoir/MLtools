import numpy as np
from env_settting import SKLEARN_PARAMS_SAVE_PATH
from script.data_handler.Base.BaseDataset import BaseDataset
from script.sklearn_like_toolkit.param_search.HyperOpt.HyperOpt import HyperOpt, HyperOpt_fn
from script.sklearn_like_toolkit.warpper.base.MixIn import DFEncoderMixIn, YLabelOneHotConvertMixIn
from script.sklearn_like_toolkit.warpper.wrapperGridSearchCV import wrapperGridSearchCV
from script.util.MixIn import LoggerMixIn


class SupervisedHyperOptCVFunc(HyperOpt_fn):
    @staticmethod
    def fn(params, feed_args, feed_kwargs):
        clf_cls = feed_kwargs['clf_cls']
        dataset = feed_kwargs['dataset']
        cv = feed_kwargs['cv']
        metric = feed_kwargs['metric']
        split_rate = feed_kwargs['split_rate']

        scores = []
        for i in range(cv):
            dataset.shuffle()
            train_set, test_set = dataset.split(ratio=split_rate)
            train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
            test_Xs, test_Ys = test_set.full_batch(['Xs', 'Ys'])

            clf = clf_cls(**params)
            clf.fit(train_Xs, train_Ys)
            score = clf.score(test_Xs, test_Ys)
            scores += [score]

        return np.mean(scores)


class ParamOpt(DFEncoderMixIn, YLabelOneHotConvertMixIn, LoggerMixIn):
    """
    Hyperopt
    grid_serach
    random_search
    """

    def __init__(
            self, func=None, cv=3, n_iter=None, opt_method='HyperOpt', min_best=True,
            n_job=1, x_df_encoder=None, y_df_encoder=None, clone=True, verbose=0):
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
            'GridSearch': self.gridSearchCV
        }
        self.n_iter = n_iter
        self.min_best = min_best
        self.n_jobs = n_job
        self.clone = clone

        self.optimizers = {}
        self.optimize_result = {}
        self.params_save_path = SKLEARN_PARAMS_SAVE_PATH

    def _refit(self, clf, x, y, **kwargs):
        dataset = BaseDataset(x=x, y=y)
        dataset.shuffle()
        train, _ = dataset.split()
        train_x, train_y = train.full_batch()

        clf.fit(train_x, train_y)
        return clf

    def HyperOptSearchCV(self, clf, x, y, **kwargs):
        n_iter = self.n_iter
        min_best = self.min_best
        if self.n_jobs > 1:
            parallel = True
        else:
            parallel = False

        x, y = self._if_df_encode(x, y)
        y = self.np_arr_to_index(y)
        dataset = BaseDataset(x=x, y=y)

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
                'dataset': dataset
            },
            min_best=min_best
        )
        self.optimize_result = trials
        self.optimizers = opt
        clf_optimized = clf.__class__(**opt.best_param)

        return self._refit(clf_optimized, x, y)

    def gridSearchCV(self, clf, x, y, **kwargs):
        x, y = self._if_df_encode(x, y)
        y = self.np_arr_to_index(y)

        optimizer = wrapperGridSearchCV(clf, clf.tuning_grid, scoring=None, n_jobs=self.n_jobs, cv=self.cv)
        optimizer.fit(x, y)

        self.optimize_result = optimizer.cv_results_
        return optimizer.best_estimator_

    def _fit(self, clf, x, y, **kwargs):
        x, y = self._if_df_encode(x, y)
        y = self.np_arr_to_index(y)

        opt_func = self.opt_funcs[self.opt_method]
        clf_optimized = opt_func(clf, x, y, *kwargs)

        return clf_optimized

        # def fit_pack(self, clf_pack, x, y, **kwargs):

    #     x, y = self._if_df_encode(x, y)
    #     y = self.np_arr_to_index(y)
    #
    #     opt_func = self.opt_funcs[self.opt_method]
    #     pack = clf_pack.pack
    #
    #     params_pack = {}
    #     for clf_name, clf_ in clf.pack.items():
    #         print(f'optimize {clf_name}')
    #         clf_optimized = opt_func(clf, x, y, *kwargs)
    #         params_pack[clf_name] = clf_optimized.get_paramsa()

    def fit(self, clf, x, y, **kwargs):
        return self._fit(clf, x, y, **kwargs)
