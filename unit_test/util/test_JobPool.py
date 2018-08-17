import time
from script.util.JobPool import JobPool


def func(*args, **kwargs):
    print(args, kwargs)
    time.sleep(2)


def test_JobPool():
    args = [1, 2, 3]
    kwargs = {"a": 1, "b": 2, "c": 3}
    with JobPool() as pool:
        for i in range(10):
            pool.apply_async(func, args=args, kwargs=kwargs)



