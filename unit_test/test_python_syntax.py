from pprint import pprint


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
