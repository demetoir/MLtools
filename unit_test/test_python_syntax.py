from pprint import pprint
from functools import wraps
import pickle
import dill

from script.util.misc_util import log_error_trace


def test_metaclass_mixIn():
    class meta_B(type):

        def __init__(cls, name, bases, cls_dict):
            type.__init__(cls, name, bases, cls_dict)
            print('meta_B')

    class meta_A(type):

        def __init__(cls, name, bases, cls_dict):
            type.__init__(cls, name, bases, cls_dict)
            print('meta_A')

    class A(metaclass=meta_A):
        def __init__(self):
            print('class_A')

    class B(metaclass=meta_B):
        def __init__(self):
            print('class_B')

    # noinspection PyMethodParameters
    class meta_AB(meta_A, meta_B):
        def __init__(cls, name, bases, cls_dict):
            meta_A.__init__(cls, name, bases, cls_dict)
            meta_B.__init__(cls, name, bases, cls_dict)
            print('meta_AB')

        pass

    class C(A, B, metaclass=meta_AB):
        def __init__(self):
            A.__init__(self)
            B.__init__(self)

    A()
    B()
    C()


def test_getitem():
    class cls:
        def __init__(self):
            self.pack = [i for i in range(10)]

        def __getitem__(self, item):
            return self.pack.__getitem__(item)

    c = cls()
    for i in range(10):
        print(c[i])
    print(c[3:5])


def test_multi_inherite_super_init_arg():
    class A:
        def __init__(self, arg1, arg2, **kwargs):
            print('A.__init')
            self.arg1 = arg1
            self.arg2 = arg2
            self.kwarg1 = None

    class B:
        def __init__(self, arg3, arg4, **kwargs):
            print('B.__init')
            self.arg3 = arg3
            self.arg4 = arg4
            self.kwarg2 = None

    class C(A, B):
        def __init__(self, arg1, arg2, arg3, arg4, arg5, **kwargs):
            print('C.__init')
            A.__init__(self, arg1, arg2)
            B.__init__(self, arg3, arg4)
            self.arg5 = arg5
            self.kwarg3 = kwargs

    a = A(1, 2)
    b = B(3, 4)
    c = C(1, 2, 3, 4, 5)

    pprint(a.__dict__)
    pprint(b.__dict__)
    pprint(c.__dict__)


def deco(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # pprint(args)
        # pprint(kwargs)
        ret = func(*args, **kwargs)
        return ret

    return wrapper


@deco
def define_with_deco(*args, **kwargs):
    pprint('define_with_deco')


def closure(*args, **kwargs):
    pprint('closure')


def dump_and_load(pickler, func):
    try:
        path = './pkl'
        with open(path, 'wb') as f:
            pickler.dump(func, f)
        with open(path, 'rb') as f:
            func = pickler.load(f)

        func()
    except BaseException as e:
        log_error_trace(pprint, e)


def test_pickle_deco_and_closure():
    def local_closure():
        pprint('local closure')

    closure_with_deco = deco(closure)

    local_closure_with_deco = deco(local_closure)

    print('pickle')
    dump_and_load(pickle, closure_with_deco)
    dump_and_load(pickle, define_with_deco)
    dump_and_load(pickle, local_closure_with_deco)
    print('dill')
    dump_and_load(dill, closure_with_deco)
    dump_and_load(dill, define_with_deco)
    dump_and_load(dill, local_closure_with_deco)
