from script.sklearn_like_toolkit.base.MixIn import ClfWrapperMixIn


class BaseWrapperClf(ClfWrapperMixIn):
    def __init__(self):
        ClfWrapperMixIn.__init__(self)
