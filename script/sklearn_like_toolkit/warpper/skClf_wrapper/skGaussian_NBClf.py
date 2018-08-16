from sklearn.naive_bayes import GaussianNB as _skGaussianNB

from script.sklearn_like_toolkit.warpper.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.warpper.base.MixIn import MetaBaseWrapperClf_with_ABC


class skGaussian_NBClf(BaseWrapperClf, _skGaussianNB, metaclass=MetaBaseWrapperClf_with_ABC):
    def __init__(self, priors=None):
        _skGaussianNB.__init__(self, priors)
        BaseWrapperClf.__init__(self)

    HyperOpt_space = {}
    tuning_grid = {}
    tuning_params = {
        'priors': None
    }
