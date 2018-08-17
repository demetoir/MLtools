from sklearn.isotonic import IsotonicRegression as _IsotonicRegression

from script.sklearn_like_toolkit.warpper.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.warpper.base.MixIn import MetaBaseWrapperRegWithABC


class skIsotonicReg(_IsotonicRegression, BaseWrapperReg, metaclass=MetaBaseWrapperRegWithABC):

    def __init__(self, y_min=None, y_max=None, increasing=True, out_of_bounds='nan'):
        _IsotonicRegression.__init__(self, y_min, y_max, increasing, out_of_bounds)
        BaseWrapperReg.__init__(self)

    HyperOpt_space = {}

    tuning_grid = {
        'y_min': None,
        'y_max': None,
        'increasing': True,
        'out_of_bounds': 'nan',
    }
