from script.util.misc_util import *
import traceback
import sys
import numpy as np
import sklearn.utils
import pandas as pd
from script.util.MixIn import LoggerMixIn, PickleMixIn
from script.util.numpy_utils import reformat_np_arr


class MetaDataset(type):
    """Metaclass for hook inherited class's function
    metaclass ref from 'https://code.i-harness.com/ko/q/11fc307'
    """

    def __init__(cls, name, bases, cls_dict):
        type.__init__(cls, name, bases, cls_dict)

        # hook  after_load for BaseDataset.load
        new_load = None
        if 'load' in cls_dict:
            def new_load(self, path, **kwargs):
                try:
                    func = cls_dict['load']
                    func(self, path, **kwargs)
                    self._after_load()
                except Exception:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    err_msg = traceback.format_exception(exc_type, exc_value, exc_traceback)
                    self.log.error(*err_msg)

        setattr(cls, 'load', new_load)


class BaseDataset(LoggerMixIn, PickleMixIn, metaclass=MetaDataset):

    def __init__(self, x=None, y=None, base='DataFrame', verbose=20, caching=True, with_id=True, **kwargs):
        LoggerMixIn.__init__(self, verbose)
        self.caching = caching
        self.with_id = with_id
        self.base = base

        self._data = {}

        if isinstance(x, np.ndarray):
            self._from_np_x(x)
        elif isinstance(x, pd.DataFrame):
            self._from_x_df(x)

        if isinstance(y, np.ndarray):
            self._from_np_y(y)
        elif isinstance(y, pd.DataFrame):
            self._from_y_df(y)

        self.cursor = 0
        self._cursor_group_by_class = None
        self._idxs_group_by_label = None
        self._n_classes = None
        self._classes = None
        self._size_group_by_class = None

        self.kwargs = kwargs

    def __str__(self):
        __str__ = f"{self.__class__.__name__}\n" \
                  f"size = {self.size}\n" \
                  f"keys = {self.keys}\n" \
                  f"x_keys = {self.x_keys}\n"
        if self.y_keys:
            __str__ += f"y_keys = {self.y_keys}\n" \
                       f"classes = {self.classes}\n" \
                       f"size group by class = {self.size_group_by_class}\n"
        return __str__

    def __repr__(self):
        __str__ = f"{self.__class__.__name__}\n" \
                  f"size = {self.size}\n" \
                  f"keys = {self.keys}\n" \
                  f"x_keys = {self.x_keys}\n"
        if self.y_keys:
            __str__ += f"y_keys = {self.y_keys}\n" \
                       f"classes = {self.classes}\n" \
                       f"size group by class = {self.size_group_by_class}\n"
        return __str__

    def __getitem__(self, item):
        return self._data.__getitem__(item)

    def __setitem__(self, key, value):
        return self._data.__setitem__(key, np.array(value))

    @property
    def data(self):
        return self._data

    @property
    def classes(self):
        if self._classes is None:
            if self.y_label is not None:
                self._classes = sorted(list(np.unique(self.y_label)))

        return self._classes

    @property
    def n_classes(self):
        if self._n_classes is None:
            self._n_classes = len(self.classes)

        return self._n_classes

    @property
    def size(self):
        return max([len(val) for key, val in self.data.items()])

    @property
    def size_group_by_class(self):
        if self._size_group_by_class is None:
            self._size_group_by_class = {
                class_: len(self.idxs_group_by_class[class_])
                for class_ in self.idxs_group_by_class
            }

        return self._size_group_by_class

    @property
    def idxs_group_by_class(self):
        if self._idxs_group_by_label is None:
            self._idxs_group_by_label = {
                class_: np.where(self.y_label == class_)[0]
                for class_ in self.classes
            }

        return self._idxs_group_by_label

    @property
    def input_shapes(self):
        return {key: list(self.data[key].shape[1:]) for key in self.data}

    @property
    def y(self):
        if 'y' in self._data:
            return self._data['y']
        elif self.y_keys:
            if len(self.y_keys) == 1:
                y = self.data[self.y_keys[0]]
            else:
                y = {
                    key: self.data[key]
                    for key in self.y_keys
                }
        else:
            y = None

        return y

    @property
    def x(self):
        if 'x' in self._data:
            return self._data['x']

        if self.x_keys is None:
            keys = self.keys
            return {
                key: self.data[key]
                for key in keys
            }
        else:
            keys = self.x_keys

            return {
                key: self.data[key]
                for key in keys
            }

    @property
    def y_label(self):
        if self.y is None:
            return None
        else:
            return reformat_np_arr(self.y, 'index')

    @property
    def y_onehot(self):
        if self.y is None:
            return None
        else:
            return reformat_np_arr(self.y, 'onehot')

    @property
    def keys(self):
        return self._data.keys()

    @property
    def cursor_group_by_class(self):
        if self._cursor_group_by_class is None:
            self._cursor_group_by_class = {
                class_: 0
                for class_ in self.classes
            }

        return self._cursor_group_by_class

    def _clone(self, **kwargs):
        return self.__class__(**kwargs)

    def _invalidate(self):
        self._cursor_group_by_class = None
        self._idxs_group_by_label = None
        self._n_classes = None
        self._classes = None
        self._size_group_by_class = None
        self.reset_id()

    def _iter_batch(self, data, batch_size, cursor=None):
        if batch_size == 0:
            raise ValueError(f'batch size > 0')

        if cursor is None:
            cursor = self.cursor

        data_size = len(data)

        batch_to_append = None
        if batch_size >= data_size:
            batch_to_append = np.repeat(data, batch_size // data_size, axis=0)
            batch_size = batch_size % data_size

        if batch_size > 0:
            begin, end = cursor, (cursor + batch_size) % data_size

            if begin < end:
                batch = data[begin:end]
            else:
                first, second = data[begin:], data[:end]
                batch = np.concatenate((first, second))

                if batch_to_append is not None:
                    batch = np.concatenate((batch, batch_to_append))
        else:
            batch = batch_to_append

        return batch

    def _after_load(self):
        if self.with_id:
            self.reset_id()

        self.log.info(f'{self.__class__.__name__} {self.size} rows loaded')

    def _iter_batch_balanced(self, key, batch_size_group_by_class):
        batch = [
            self._iter_batch(
                self.data[key][self.idxs_group_by_class[class_]],
                batch_size_group_by_class[class_],
                self.cursor_group_by_class[class_]
            )
            for class_ in self.classes
        ]
        return np.concatenate(batch, axis=0)

    @staticmethod
    def _concat_batch(batch):
        feature_size = len(batch)
        batch_size = len(list(batch.values())[0])

        if feature_size == 1:
            batch = list(batch.values())[0]
        else:
            batch = batch.items()

            batch = np.vstack(batch)
            batch = batch.reshape([batch_size, -1])

        if batch.ndim >= 2 and batch.shape[1] == 1:
            batch = batch.reshape([batch_size])

        return batch

    def _update_cursor(self, size):
        self.cursor = (self.cursor + size) % self.size

    def _update_cursor_group_by_class(self, size):
        div = size // self.n_classes
        batch_size_group_by_class = {key: div for key in self.classes}
        for class_ in self.classes:
            self._cursor_group_by_class[class_] += batch_size_group_by_class[class_]
            self._cursor_group_by_class[class_] %= self.size_group_by_class[class_]

    def update_cursor(self, size, balanced_class=False):
        if balanced_class:
            self._update_cursor_group_by_class(size)
        else:
            self._update_cursor(size)

    def _collect_iter_batch(self, keys, size, balanced_class=False, out_type=None):
        if balanced_class:
            div = size // self.n_classes
            batch_size_group_by_class = {key: div for key in self.classes}
            plus_one_idx_key = np.random.choice(self.classes, size % self.n_classes, replace=False)
            for key in plus_one_idx_key:
                batch_size_group_by_class[key] += 1

            batch = {
                key: self._iter_batch_balanced(key, batch_size_group_by_class)
                for key in keys
            }
        else:
            batch = {
                key: self._iter_batch(self.data[key], size)
                for key in keys
            }

        batch = self._batch_convert(batch, out_type)

        return batch

    def _from_x_df(self, x_df):
        for key in x_df:
            self.add_data(key, x_df[key])
        self.x_keys = list(x_df.columns)

    def _from_y_df(self, y_df):
        for key in y_df:
            self.add_data(key, y_df[key])
        self.y_keys = list(y_df.columns)

    def _from_np_x(self, np_x):
        self.add_data('x', np_x)
        self.x_keys = ['x']

    def _from_np_y(self, np_y):
        self.add_data('y', np_y)
        self.y_keys = ['y']

    def add_data(self, key, data):
        self._data[key] = np.array(data)

    def append_data(self, key, data):
        if key not in self._data:
            self._data[key] = np.array(data)
        else:
            self._data[key] = np.concatenate((self._data[key], data))

    def reset_id(self):
        if 'id_' not in self._data.keys():
            self._data['id_'] = np.array([i for i in range(1, self.size + 1)]).reshape([self.size, 1])

    def load(self, path):
        """load dataset from file should implement

        save data at self.data, expect dict type

        :param path:
        :return: None
        """
        raise NotImplementedError

    def _batch_convert(self, batch, out_type):
        self.convert_out_type_func = {
            'concat': self._concat_batch,
            'DataFrame': pd.DataFrame,
            'df': pd.DataFrame,
            'np_dict': lambda x: x
        }
        convert_func = self.convert_out_type_func[out_type]
        return convert_func(batch)

    def next_batch(self, batch_size, batch_keys=None, update_cursor=True, balanced_class=False, out_type='concat'):
        if type(batch_keys) is str:
            batch_keys = [batch_keys]

        if batch_keys is None:
            x = self._collect_iter_batch(
                self.x_keys,
                batch_size,
                balanced_class,
                out_type=out_type
            )

            y = None
            if self.y_keys is not None:
                y = self._collect_iter_batch(
                    self.y_keys,
                    batch_size,
                    balanced_class,
                    out_type=out_type
                )

            batch = x if y is None else (x, y)
        else:
            batch = self._collect_iter_batch(
                batch_keys,
                batch_size,
                balanced_class,
                out_type=out_type
            )

        if update_cursor:
            self.update_cursor(batch_size, balanced_class)

        return batch

    def full_batch(self, batch_keys=None, out_type='concat'):
        return self.next_batch(self.size, batch_keys, out_type=out_type)

    def split(self, ratio=(7, 3), shuffle=False, random_state=None, balanced_class=True):
        if shuffle:
            self.shuffle(random_state)

        a_set = self._clone()
        b_set = self._clone()

        a_ratio = ratio[0] / sum(ratio)
        b_ratio = ratio[0] / sum(ratio)

        if balanced_class:
            for key in self.keys:

                a_data = []
                b_data = []
                for class_ in self.classes:
                    idxs = self.idxs_group_by_class[class_]

                    a_size = int(self.size_group_by_class[class_] * a_ratio)
                    b_size = int(self.size_group_by_class[class_] * b_ratio)
                    if a_size <= 0 or b_size <= 0:
                        raise ValueError(f'{class_} can not balanece class split, '
                                         f'total_size = {self.size_group_by_class[class_]}, '
                                         f'a_size = {a_size}, '
                                         f'b_size = {b_size}')

                    a_idxs = idxs[:a_size]
                    b_idxs = idxs[a_size:]
                    a_data += [self.data[key][a_idxs]]
                    b_data += [self.data[key][b_idxs]]

                a_data = np.concatenate(a_data)
                b_data = np.concatenate(b_data)
                a_set.add_data(key, a_data)
                b_set.add_data(key, b_data)

        else:
            index = int(self.size * a_ratio)
            for key in self._data:
                a_set.add_data(key, self._data[key][:index])
                b_set.add_data(key, self._data[key][index:])

        if shuffle:
            self.sort()

        a_set.x_keys = self.x_keys
        b_set.x_keys = self.x_keys

        a_set.y_keys = self.y_keys
        b_set.y_keys = self.y_keys
        a_set._invalidate()
        b_set._invalidate()

        return a_set, b_set

    def merge(self, a_set, b_set):
        if set(a_set.data.key()) is set(b_set.data.keys()):
            raise KeyError("dataset can not merge, key does not match")

        new_set = self._clone()
        for key in a_set.batch_keys:
            concat = np.concatenate((a_set.data[key], b_set.data[key]), axis=0)
            new_set.add_data(key, concat)

        new_set._invalidate()

        return new_set

    def shuffle(self, random_state=None):
        if random_state is None:
            random_state = np.random.randint(1, 12345678)

        for key in self._data:
            self._data[key] = sklearn.utils.shuffle(self._data[key], random_state=random_state)

        self._invalidate()

    def sort(self, sort_key=None):
        if sort_key is None:
            sort_key = 'id_'

        for key in self._data:
            if key is sort_key:
                continue

            zipped = list(zip(self._data[sort_key], self._data[key]))
            data = sorted(zipped, key=lambda x: x[0])
            a, data = zip(*data)
            self._data[key] = np.array(data)

        self._data[sort_key] = np.array(sorted(self._data[sort_key]))

        self._invalidate()

    def _to_DataFrame(self, keys):
        df = pd.DataFrame({})
        for key in keys:
            try:
                shape = self.data[key].shape

                if self.data[key].ndim > 1:
                    data = [str(i) for i in self.data[key]]
                else:
                    data = self.data[key].reshape([shape[0]])

                df[key] = data
            except BaseException as e:
                log_error_trace(self.log.warn, e)

        return df

    def to_DataFrame(self, keys=None, id_=False):
        if keys is None:
            keys = list(self._data.keys())
            if not id_:
                keys.remove('id_')

            return self._to_DataFrame(keys)
        else:
            x_df = self._to_DataFrame(self.x_keys)
            if self.y_keys:
                y_df = self._to_DataFrame(self.y_keys)
                return x_df, y_df
            else:
                return x_df

    def from_DataFrame(self, x_df, y_df=None):
        obj = self._clone()
        obj._from_x_df(x_df)
        if y_df is not None:
            obj._from_y_df(y_df)

        return obj

    def to_dummyDataset(self, keys=None):
        dataset = BaseDataset()

        if keys is None:
            keys = self._data.keys()

        for key in keys:
            dataset.add_data(key, self._data[key])

        return dataset

    def to_np_arr(self):
        return self.full_batch()

    def from_np_arr(self, x_np, y_np=None):
        obj = self._clone()
        obj._from_np_x(x_np)
        if y_np is not None:
            obj._from_np_y(y_np)
        return obj
