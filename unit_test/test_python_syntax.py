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
