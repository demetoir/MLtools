from script.data_handler.Base.BaseDataset import BaseDataset


class DummyDataset(BaseDataset):
    def load(self, path):
        pass

    def save(self):
        pass

    def transform(self):
        pass
