from sklearn.ensemble import VotingClassifier as _skVotingClassifier

from script.sklearn_like_toolkit.warpper.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.warpper.base.MixIn import MetaBaseWrapperClfWithABC


class skVotingClf(BaseWrapperClf, _skVotingClassifier, metaclass=MetaBaseWrapperClfWithABC):
    def __init__(self, estimators, voting='hard', weights=None, n_jobs=1, flatten_transform=None):
        _skVotingClassifier.__init__(self, estimators, voting, weights, n_jobs, flatten_transform)
        BaseWrapperClf.__init__(self)

    HyperOpt_space = {}

    tuning_grid = {}

    tuning_params = {}
