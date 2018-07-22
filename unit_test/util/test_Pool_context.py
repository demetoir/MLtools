import time
from script.util.JobPool import JobPool


def test_func(*args, **kwargs):
    print(args, kwargs)
    time.sleep(3)


def test_pool_context():
    args = [1, 2, 3]
    kwargs = {"a": 1, "b": 2, "c": 3}
    with JobPool() as pool:
        for i in range(10):
            pool.apply_async(test_func, args=args, kwargs=kwargs)
