import multiprocessing as mp
import tqdm
from script.util.IdDict import IdDict
from multiprocessing_on_dill.pool import Pool as _Pool
from multiprocessing_on_dill.context import TimeoutError
from queue import Queue

CPU_COUNT = mp.cpu_count() - 1


class Pool_context:
    _singleton_pool = None

    @property
    def pool(self):
        if self.__class__._singleton_pool is None:
            self.__class__._singleton_pool = _Pool(processes=self.processes)

        return self.__class__._singleton_pool

    def __init__(self, processes=CPU_COUNT) -> None:
        super().__init__()
        self.processes = processes
        self.childs = IdDict()
        self.tic = 0.01
        self._id_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("waiting process")

        pbar = tqdm.tqdm(total=len(self.childs))

        q = Queue()
        for id_, child in self.childs.items():
            q.put((id_, child))

        while q.qsize() > 0:
            id_, child = q.get()
            try:
                child.get(self.tic)
                pbar.update(1)
            except TimeoutError:
                q.put((id_, child))

        pbar.close()

    def put_child(self, child):
        return self.childs.put(child)

    def get_child(self, id_):
        return self.childs[id_]

    def apply(self, func, args=None, kwargs=None):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        child = self.pool.apply_async(func=func, args=args, kwds=kwargs)
        return child.get()

    def apply_async(self, func, args=None, kwargs=None):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        child = self.pool.apply_async(func=func, args=args, kwds=kwargs)
        id_ = self.childs.put(child)
        return id_

    def get(self, child_id, timeout=None):
        child = self.childs[child_id]
        return child.get(timeout)
