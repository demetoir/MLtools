class IdDict:
    def __init__(self):
        self.dict = {}
        self._id = 0

    @property
    def new_id(self):
        self._id += 1
        return self._id

    def put(self, x):
        id_ = self.new_id
        self.dict[id_] = x
        return id_

    def get(self, id_):
        return self.dict[id_]

    def pop(self, id_):
        return self.dict.pop(id_)

    def __iter__(self):
        return self.dict.__iter__()

    def items(self):
        return self.dict.items()

    def __getitem__(self, item):
        return self.dict.__getitem__(item)

    def __len__(self):
        return len(self.dict)
