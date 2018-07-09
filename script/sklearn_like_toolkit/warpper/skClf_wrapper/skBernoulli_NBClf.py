from hyperopt import hp
from sklearn.naive_bayes import BernoulliNB as _skBernoulliNB

from script.sklearn_like_toolkit.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.base.MixIn import meta_BaseWrapperClf_with_ABC


class skBernoulli_NBClf(BaseWrapperClf, _skBernoulliNB, metaclass=meta_BaseWrapperClf_with_ABC):
    def __init__(self, alpha=1.0, binarize=.0, fit_prior=True, class_prior=None):
        _skBernoulliNB.__init__(self, alpha, binarize, fit_prior, class_prior)
        BaseWrapperClf.__init__(self)

    HyperOpt_space = {
        'alpha': hp.loguniform('alpha', -8, 1),
        'binarize': hp.uniform('binarize', 0, 1)
    }
    tuning_grid = {
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        'binarize': [i / 10.0 for i in range(0, 10)],
        # 'class_prior': None,
        # 'fit_prior': True
    }
