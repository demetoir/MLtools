import multiprocessing as mp
import tqdm
from script.util.IdDict import IdDict
from multiprocessing_on_dill.pool import Pool as _Pool
from multiprocessing_on_dill.context import TimeoutError
from queue import Queue

from script.util.MixIn import LoggerMixIn
from script.util.misc_util import log_error_trace

CPU_COUNT = mp.cpu_count() - 1


def warmup_fn():
    return [i for i in range(5)]


class JobPool(LoggerMixIn):
    _singleton_pool = None
    TIMEOUT = 0.01

    def __init__(self, processes=CPU_COUNT, verbose=30) -> None:
        super().__init__(verbose)
        self.processes = processes
        self.childs = IdDict()

        self._id_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("join process")
        self.join()

    @property
    def pool(self):
        if self.__class__._singleton_pool is None:
            self.__class__._singleton_pool = _Pool(processes=self.processes)

        return self.__class__._singleton_pool

    def join(self):
        pbar = tqdm.tqdm(total=len(self.childs))

        q = Queue()
        for id_, child in self.childs.items():
            q.put((id_, child))

        while q.qsize() > 0:
            id_, child = q.get()
            try:
                child.get(self.TIMEOUT)
                pbar.update(1)
            except TimeoutError:
                q.put((id_, child))
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except BaseException as e:
                pbar.update(1)
                self.log.warn(f'job fail')
                log_error_trace(self.log.info, e)

        pbar.close()

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

    def put_child(self, child):
        return self.childs.put(child)

    def get_child(self, id_):
        return self.childs[id_]
