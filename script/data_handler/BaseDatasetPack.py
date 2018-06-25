from script.util.Logger import Logger


class BaseDatasetPack:
    def __init__(self, caching=True, **kwargs):
        self.logger = Logger(self.__class__.__name__)
        self.log = self.logger.get_log()
        self.pack = {}
        self.caching = caching

    def load(self, path, **kwargs):
        for k in self.pack:
            self.pack[k].load(path, **kwargs)

    def shuffle(self):
        for key in self.pack:
            self.pack[key].shuffle()

    def split(self, from_key, a_key, b_key, rate):
        from_set = self.pack[from_key]
        self.pack.pop(from_key)

        a_set, b_set = from_set.split(rate)
        self.pack[a_key] = a_set
        self.pack[b_key] = b_set
        return a_set, b_set

    def merge_shuffle(self, a_key, b_key, rate):
        a_set = self.pack[a_key]
        b_set = self.pack[b_key]

        merge_set = a_set.merge(a_set, b_set)
        merge_set.shuffle()
        a_set, b_set = merge_set.split(rate, shuffle=True)

        self.pack[a_key] = a_set
        self.pack[b_key] = b_set
        return a_set, b_set

    def merge(self, a_key, b_key, merge_set_key):
        a_set = self.pack[a_key]
        b_set = self.pack[b_key]
        self.pack.pop(a_key)
        self.pack.pop(b_key)

        merge_set = a_set.merge(a_set, b_set)
        self.pack[merge_set_key] = merge_set

    def sort(self, sort_key=None):
        for key in self.pack:
            self.pack[key].sort(sort_key)

    def __getitem__(self, item):
        return self.pack.__getitem__(item)
