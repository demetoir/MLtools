from script.sklearn_like_toolkit.base.MixIn import ClfWrapperMixIn


class BaseWrapperClf(ClfWrapperMixIn):
    def __init__(self):
        ClfWrapperMixIn.__init__(self)

    def fit(self, Xs, Ys, **kwargs):
        return getattr(super(), 'fit')(Xs, Ys, **kwargs)

    def score(self, Xs, Ys, **kwargs):
        return getattr(super(), 'score')(Xs, Ys, **kwargs)

    def predict(self, Xs, **kwargs):
        return getattr(super(), 'predict')(Xs, **kwargs)

    def predict_proba(self, Xs, **kwargs):
        return getattr(super(), 'predict_proba')(Xs, **kwargs)
