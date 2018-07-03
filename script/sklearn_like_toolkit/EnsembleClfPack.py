from script.sklearn_like_toolkit.base.BaseWrapperClfPack import BaseWrapperClfPack
from script.sklearn_like_toolkit.FoldingHardVoteClf import FoldingHardVoteClf
from script.sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxStackingCVClf, mlxStackingClf


class EnsembleClfPack(BaseWrapperClfPack):
    class_pack = {
        'FoldingHardVote': FoldingHardVoteClf,
        'mlxStackingCVClf': mlxStackingCVClf,
        'mlxStackingClf': mlxStackingClf
    }

    def __init__(self, clfs, meta_clf=None):
        super().__init__()

        self.pack = {
            'FoldingHardVote': FoldingHardVoteClf(clfs),
            'mlxStackingCVClf': mlxStackingCVClf(clfs, meta_clf),
            'mlxStackingClf': mlxStackingClf(clfs, meta_clf)
        }
