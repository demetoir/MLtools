from env_settting import ROOT_PATH
from unit_test.test_helper.dummy.DummyModel import DummyModel


class test_AbstractModel:
    def __init__(self):
        self.root_path = ROOT_PATH
        self.DummyModel = DummyModel
        pass

    def test__00(self):
        DummyModel(ROOT_PATH)
