from hyperopt import hp
from sklearn.naive_bayes import MultinomialNB as _skMultinomialNB

from script.sklearn_like_toolkit.warpper.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.warpper.base.MixIn import MetaBaseWrapperClf_with_ABC


class skMultinomial_NBClf(BaseWrapperClf, _skMultinomialNB, metaclass=MetaBaseWrapperClf_with_ABC):
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        _skMultinomialNB.__init__(self, alpha, fit_prior, class_prior)
        BaseWrapperClf.__init__(self)

    HyperOpt_space = {
        'alpha': hp.loguniform('alpha', -8, 1),
    }

    tuning_grid = {
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        # 'class_prior': None,
        # 'fit_prior': True
    }
