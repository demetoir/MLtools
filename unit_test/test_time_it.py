from util.deco import deco_timeit


@deco_timeit
def assign(n):
    a = [i for i in range(100)]
    for _ in range(n):
        idx = [i for i in range(50)]
        a = idx


@deco_timeit
def direct(n):
    a = [i for i in range(100)]
    for _ in range(n):
        a = [i for i in range(50)]


def test_timing_assign_vs_direct():
    n = 1000000
    assign(n)
    direct(n)
