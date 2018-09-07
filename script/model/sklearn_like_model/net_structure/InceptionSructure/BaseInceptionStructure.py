from script.model.sklearn_like_model.net_structure.Base_net_structure import Base_net_structure


class BaseInceptionStructure(Base_net_structure):
    def __init__(self, x, n_classes, reuse=False, name=None, verbose=0):
        super().__init__(reuse, name, verbose)
        self.x = x
        self.n_classes = n_classes

    def build(self):
        raise NotImplementedError
