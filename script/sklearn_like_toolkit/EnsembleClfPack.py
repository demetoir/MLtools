from script.sklearn_like_toolkit.base.BaseWrapperPack import BaseWrapperPack
from script.sklearn_like_toolkit.FoldingHardVote import FoldingHardVote
from script.sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxStackingCVClf, mlxStackingClf


class EnsembleClfPack(BaseWrapperPack):
    class_pack = {
        'FoldingHardVote': FoldingHardVote,
        'mlxStackingCVClf': mlxStackingCVClf,
        'mlxStackingClf': mlxStackingClf
    }

    def __init__(self, clfs, meta_clf=None):
        super().__init__()

        self.pack = {
            'FoldingHardVote': FoldingHardVote(clfs),
            'mlxStackingCVClf': mlxStackingCVClf(clfs, meta_clf),
            'mlxStackingClf': mlxStackingClf(clfs, meta_clf)
        }
