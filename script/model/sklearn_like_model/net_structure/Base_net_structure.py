from script.util.MixIn import LoggerMixIn
from script.util.tensor_ops import collect_vars, join_scope, get_scope


class Base_net_structure(LoggerMixIn):
    def __init__(self, reuse=False, name='Base_net_structure', verbose=0):
        super().__init__(verbose)
        self.reuse = reuse
        self.name = name
        self._vars = None

    @property
    def vars(self):
        if self._vars is None:
            if len(get_scope()) == 0:
                self._vars = collect_vars(self.name)
            else:
                self._vars = collect_vars(join_scope(get_scope(), self.name))
        return self._vars

    def build(self):
        raise NotImplementedError
