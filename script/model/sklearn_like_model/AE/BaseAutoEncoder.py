from script.model.sklearn_like_model.BaseModel import BaseModel


class basicAEPropertyMixIn:

    @property
    def _Xs(self):
        return getattr(self, 'Xs')

    @property
    def _zs(self):
        return getattr(self, 'zs')

    @property
    def _train_ops(self):
        return [
            getattr(self, 'train_op'),
            getattr(self, 'op_inc_global_step')
        ]

    @property
    def _code_ops(self):
        return getattr(self, 'latent_code')

    @property
    def _recon_ops(self):
        return getattr(self, 'Xs_recon')

    @property
    def _generate_ops(self):
        return getattr(self, 'Xs_gen')

    @property
    def _metric_ops(self):
        return getattr(self, 'loss')

    @property
    def _noise(self):
        return getattr(self, 'noise')


class BaseAutoEncoder(BaseModel, basicAEPropertyMixIn):
    def __init__(self, verbose=10, **kwargs):
        super().__init__(verbose, **kwargs)
        self.batch_size = None

    def _build_input_shapes(self, input_shapes):
        raise NotImplementedError

    def _build_main_graph(self):
        raise NotImplementedError

    def _build_loss_function(self):
        raise NotImplementedError

    def _build_train_ops(self):
        raise NotImplementedError

    def code(self, Xs):
        return self.get_tf_values(self._code_ops, {self._Xs: Xs})

    def recon(self, Xs):
        return self.get_tf_values(self._recon_ops, {self._Xs: Xs})

    def metric(self, Xs):
        return self.get_tf_values(self._metric_ops, {self._Xs: Xs})

    def generate(self, zs):
        return self.get_tf_values(self._recon_ops, {self._zs: zs})

    def random_z(self):
        raise NotImplementedError
