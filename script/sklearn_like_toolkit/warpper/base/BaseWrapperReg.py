from script.sklearn_like_toolkit.warpper.base.MixIn import RegWrapperMixIn


class BaseWrapperReg(RegWrapperMixIn):
    def __init__(self):
        RegWrapperMixIn.__init__(self)
