from script.model.sklearn_like_model.BaseModel import BaseModel
import numpy as np
from tqdm import trange


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


class basicAEOpsMIxIn:
    pass


class BaseAutoEncoder(BaseModel, basicAEPropertyMixIn):
    def __init__(self, verbose=10, **kwargs):
        super().__init__(verbose, **kwargs)
        self.batch_size = None

    def build_input_shapes(self, input_shapes):
        raise NotImplementedError

    def _build_main_graph(self):
        raise NotImplementedError

    def _build_loss_function(self):
        raise NotImplementedError

    def _build_train_ops(self):
        raise NotImplementedError

    def train(self, Xs, epoch=100, save_interval=None, batch_size=None):
        shapes = {'Xs': Xs.shape[1:]}
        self._apply_input_shapes(self.build_input_shapes(shapes))
        self.is_built()

        dataset = self.to_dummyDataset(Xs=Xs)

        if batch_size is None:
            batch_size = self.batch_size

        iter_num = 0
        iter_per_epoch = dataset.size // batch_size
        self.log.info("train epoch {}, iter/epoch {}".format(epoch, iter_per_epoch))
        for e in trange(epoch):
            dataset.shuffle()
            for i in range(iter_per_epoch):
                iter_num += 1

                Xs = dataset.next_batch(batch_size, batch_keys=['Xs'])
                self.sess.run(self._train_ops, feed_dict={self._Xs: Xs})

            Xs = dataset.next_batch(batch_size, batch_keys=['Xs'], look_up=False)
            loss = self.sess.run(self._metric_ops, feed_dict={self._Xs: Xs})
            loss = np.mean(loss)
            self.log.info("e:{e} loss : {loss}".format(e=e, loss=loss))

            if save_interval is not None and e % save_interval == 0:
                self.save()

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
