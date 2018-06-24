from sklearn_like_toolkit.base.MixIn import meta_BaseWrapperClf, ClfWrapperMixIn


class BaseWrapperClf(ClfWrapperMixIn):
    def __init__(self):
        ClfWrapperMixIn.__init__(self)
