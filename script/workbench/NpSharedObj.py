import ctypes
import numpy as np
from multiprocessing import Pool, Array

from script.util.elapse_time import elapse_time


class NpSharedObj:
    d_to_ctype = {
        np.double: ctypes.c_double,
        'float64': ctypes.c_float,
        'float32': ctypes.c_float,

        'int8': ctypes.c_int8,
        'int16': ctypes.c_int16,
        'int32': ctypes.c_int32,
        'int64': ctypes.c_int64,

        'uint8': ctypes.c_uint8,
        'uint16': ctypes.c_uint16,
        'uint32': ctypes.c_uint32,
        'uint64': ctypes.c_uint64,
    }

    def __init__(self):
        self._lock = None
        self._size = None
        self._shape = None
        self._dtype = None
        self._ctype = None
        self._loc = None
        self._np = None

    @staticmethod
    def from_np(np_arr, lock=False):
        obj = NpSharedObj()
        obj._lock = lock
        obj._size = np_arr.size
        obj._shape = np_arr.shape
        obj._dtype = np_arr.dtype
        obj._ctype = NpSharedObj.d_to_ctype[str(np_arr.dtype)]
        # obj._loc = Array(ctypes.c_double, np_arr.size, lock=lock)
        obj._loc = Array(ctypes.c_double, np_arr.size)
        obj._np = np.frombuffer(obj._loc.get_obj())
        # obj._np = np.frombuffer(obj._loc)
        obj._np[:] = np_arr.reshape([-1]).copy()
        obj._np = obj._np.reshape(obj._shape)

        return obj

    @staticmethod
    def from_loc(loc, shape, lock=False):
        obj = NpSharedObj()
        obj._lock = lock
        obj._shape = shape
        obj._loc = loc
        obj._np = NpSharedObj.new_np(obj._loc, obj._shape)
        obj._size = obj._np.size
        obj._dtype = obj._np.dtype
        obj._ctype = NpSharedObj.d_to_ctype[str(obj._dtype)]

        return obj

    @property
    def lock(self):
        return self._lock

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return self._size

    @property
    def shared_loc(self):
        return self._loc

    @property
    def np(self):
        return self._np

    @property
    def loc(self):
        return self._loc

    def encode(self):
        return self.loc, self.shape, self.lock

    @staticmethod
    def decode(x):
        loc, shape, lock = x

        return NpSharedObj.from_loc(loc, shape, lock)

    @staticmethod
    def new_np(loc, shape):
        # return np.frombuffer(loc).reshape(shape)
        return np.frombuffer(loc.get_obj()).reshape(shape)


def init_shared(loc, shape):
    global shared_dict

    shared_dict = {
        'loc': loc,
        'shape': shape,
        # 'mat': np.frombuffer(loc).reshape(shape)
        'mat': np.frombuffer(loc.get_obj()).reshape(shape)
    }


def func2(args):
    global shared_dict
    shared_mat = shared_dict['mat']
    loc = shared_dict['loc']
    shape = shared_dict['shape']

    # assign at element is fine
    shared_mat[2, 2] *= 2

    # but broadcasting is fail on parent's shared memory
    shared_mat = shared_mat + 1
    shared_mat = np.add(shared_mat, 1)
    shared_mat[0, :] += 1
    shared_mat[:] += 1

    return shared_mat


def test_NpSharedObj():
    shape = [3, 3]
    mat = np.ones(shape)
    shared_obj = NpSharedObj.from_np(mat)
    print(shared_obj.np)

    print(f'share memory')
    with Pool(processes=1, initializer=init_shared, initargs=(shared_obj.loc, shape,)) as pool:
        with elapse_time():
            childs = [pool.apply_async(func2, args=(args,)) for args in range(16)]

            for child in childs:
                a = child.get()
                print(a)

    print(f'exit')
    print(shared_obj.np)
