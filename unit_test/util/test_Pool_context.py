import time
from script.util.Pool_context import Pool_context


def test_func(*args, **kwargs):
    print(args, kwargs)
    time.sleep(3)


def test_pool_context():
    args = [1, 2, 3]
    kwargs = {"a": 1, "b": 2, "c": 3}
    with Pool_context() as pool:
        for i in range(10):
            pool.apply_async(test_func, args=args, kwargs=kwargs)
