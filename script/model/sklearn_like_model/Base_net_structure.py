from script.util.tensor_ops import collect_vars, join_scope, get_scope


class Base_net_structure:
    def __init__(self, reuse=False, name='Base_net_structure'):
        self.reuse = reuse
        self.name = name
        self._vars = None

    def get_vars(self):
        if self._vars is None:
            self._vars = collect_vars(join_scope(get_scope(), self.name))
        return self._vars

    def build(self):
        raise NotImplementedError
