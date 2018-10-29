from tqdm import trange
from script.model.sklearn_like_model.Mixin import Xs_MixIn, Ys_MixIn
from script.model.sklearn_like_model.BaseModel import BaseModel


class basicClfProperty(Xs_MixIn, Ys_MixIn):
    def __init__(self):
        Xs_MixIn.__init__(self)
        Ys_MixIn.__init__(self)

    @property
    def _predict_ops(self):
        return getattr(self, 'predict_index')

    @property
    def _score_ops(self):
        return getattr(self, 'acc_mean')

    @property
    def _proba_ops(self):
        return getattr(self, 'h')

    @property
    def _metric_ops(self):
        return getattr(self, 'loss')

    @property
    def _train_ops(self):
        return [
            getattr(self, 'train_op'),
            getattr(self, 'op_inc_global_step')
        ]


class BaseClassifierModel(BaseModel, basicClfProperty):

    def __init__(self, verbose=10, **kwargs):
        BaseModel.__init__(self, verbose, **kwargs)
        basicClfProperty.__init__(self)

        self.batch_size = None

    def _build_input_shapes(self, shapes):
        raise NotImplementedError

    def _build_main_graph(self):
        raise NotImplementedError

    def _build_loss_ops(self):
        raise NotImplementedError

    def _build_train_ops(self):
        raise NotImplementedError

    def train(self, Xs, Ys, epoch=1, save_interval=None, batch_size=None):
        self._prepare_train(Xs=Xs, Ys=Ys)
        dataset = self.to_dummyDataset(Xs=Xs, Ys=Ys)

        if batch_size is None:
            batch_size = self.batch_size

        iter_num = 0
        iter_per_epoch = dataset.size
        for e in trange(epoch):
            for i in range(iter_per_epoch):
                iter_num += 1

                Xs, Ys = dataset.next_batch(batch_size, batch_keys=['Xs', 'Ys'])
                self.sess.run(self._train_ops, feed_dict={self._Xs: Xs, self._Ys: Ys})

            Xs, Ys = dataset.next_batch(batch_size, batch_keys=['Xs', 'Ys'], update_cursor=False)
            loss = self.sess.run(self._metric_ops, feed_dict={self._Xs: Xs, self._Ys: Ys})
            self.log.info(f"e:{e}, i:{iter_num} loss : {loss}")

            if save_interval is not None and e % save_interval == 0:
                self.save()

    def predict(self, Xs):
        return self.sess.run(self._predict_ops, feed_dict={self._Xs: Xs})

    def score(self, Xs, Ys):
        return self.sess.run(self._score_ops, feed_dict={self._Xs: Xs, self._Ys: Ys})

    def predict_proba(self, Xs):
        return self.sess.run(self._proba_ops, feed_dict={self._Xs: Xs})

    def metric(self, Xs, Ys):
        return self.sess.run(self._metric_ops, feed_dict={self._Xs: Xs, self._Ys: Ys})
