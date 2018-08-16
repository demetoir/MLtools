from sklearn.kernel_ridge import KernelRidge as _KernelRidge

from script.sklearn_like_toolkit.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.base.MixIn import MetaBaseWrapperReg_with_ABC


class skKernelRidgeReg(_KernelRidge, BaseWrapperReg, metaclass=MetaBaseWrapperReg_with_ABC):

    def __init__(self, alpha=1, kernel="linear", gamma=None, degree=3, coef0=1, kernel_params=None):
        _KernelRidge.__init__(self, alpha, kernel, gamma, degree, coef0, kernel_params)
        BaseWrapperReg.__init__(self)

    HyperOpt_space = {
        # 'alpha': 1,
        # 'kernel': "linear",
        # 'gamma': None,
        # 'degree': 3,
        # 'coef0': 1,
        # 'kernel_params': None,
    }

    tuning_grid = {
        'alpha': 1,
        'kernel': "linear",
        'gamma': None,
        # 'degree': 3,
        # 'coef0': 1,
        # 'kernel_params': None,
    }