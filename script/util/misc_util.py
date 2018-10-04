"""misc utils
pickle, import module, zip, etc ..."""
import inspect
import shutil
import traceback
import webbrowser
from glob import glob
from importlib._bootstrap_external import SourceFileLoader
from time import strftime, localtime
import tarfile
import zipfile
import requests
import os
import pickle
import types
import json
import sys


def isPickleAble(obj):
    import pickle
    try:
        pickled = pickle.dumps(obj, protocol=-1)
        un_pickled_obj = pickle.loads(pickled)
        return True

    except BaseException as e:
        print(e)
        return False


def dump_pickle(obj, path):
    """dump pickle

    * [warning] use pickle module python3, python2 may incompatible

    :type path: str
    :type obj: object
    :param path: dump path
    :param obj: target data to dump
    """
    setup_file(path)

    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    """load pickle

    * [warning] use pickle module python3, python2 may incompatible

    :type path: str
    :param path: path to load data

    :return: data
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def import_class_from_module_path(path, class_name):
    """import class from module path

    :type path: str
    :type class_name: str
    :param path: path of module
    :param class_name: class name to load class from module

    :return: class

    :raise FileNotFoundError
    if path not found
    :raise AttributeError, ImportError
    """
    try:
        module_ = SourceFileLoader('', path).load_module()
        return getattr(module_, class_name)
    except FileNotFoundError:
        raise FileNotFoundError("%s not found" % path)
    except AttributeError:
        raise AttributeError("%s class not found in %s" % (class_name, path))


def import_module_from_module_path(path):
    """import module from module_path

    :type path: str
    :param path: path to import module
    :return module for
    :rtype module
    """
    return SourceFileLoader('', path).load_module()


def module_path_finder(path, name, recursive=True):
    """find module's absolute path in path

    :type path: str
    :type name: str
    :type recursive: bool
    :param path: target path to find module
    :param name: name of module
    :param recursive: option to search path recursively
    default True

    :return: module's absolute path

    :raise FileNotFoundError
    if module file not found in path
    """
    path_ = None
    paths = glob(os.path.join(path, '**', '*.py'), recursive=recursive)
    for path in paths:
        file_name = os.path.basename(path)
        if file_name == name + '.py':
            path_ = os.path.abspath(path)

    if path_ is None:
        raise FileNotFoundError("module %s not found" % name)

    return path_


def imports():
    """iter current loaded modules"""
    for name, val in globals().items():
        if isinstance(val, getattr(types, "ModuleType")):
            yield val.__name__


def dump_json(obj, path):
    """dump json

    :type obj: object
    :type path: str
    :param obj: object to dump
    :param path: path to dump
    """
    head, _ = os.path.split(path)
    setup_directory(head)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def load_json(path):
    """load json

    :type path: str
    :param path: load path
    """
    with open(path, 'r') as f:
        metadata = json.load(f)
    return metadata


def download_from_url(url, path):
    """download data from url

    :type url: str
    :type path: str
    :param url: download url
    :param path: path to save
    """

    with open(path, "wb") as f:
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(100 * dl / total_length)
                sys.stdout.write("\r[%s%s] %s%%" % ('=' * (done // 2), ' ' * (50 - (done // 2)), done))
                sys.stdout.flush()


def extract_tar(source_path, destination_path):
    """extract tar file

    :type source_path: str
    :type destination_path: str
    :param source_path:source path to extract
    :param destination_path: destination path
    """
    with tarfile.open(source_path) as file:
        file.extractall(destination_path)


def extract_zip(source_path, destination_path):
    """extract zip file

    :type source_path: str
    :type destination_path: str
    :param source_path:source path to extract
    :param destination_path: destination path
    :return:
    """
    with zipfile.ZipFile(source_path) as file:
        file.extractall(destination_path)


def extract_file(source_path, destination_path):
    """extract file zip, tar, tar.gz, ...

    :type source_path: str
    :type destination_path: str
    :param source_path:source path to extract
    :param destination_path: destination path
    """
    extender_tar = ['tar.gz', 'tar', 'gz']
    extender_zip = ['zip']

    extend = source_path.split('.')[-1]
    if extend in extender_tar:
        extract_tar(source_path, destination_path)
    elif extend in extender_zip:
        extract_zip(source_path, destination_path)


def time_stamp():
    return strftime("%Y-%m-%d_%H-%M-%S", localtime())


def check_path(path):
    # if os.path.isfile(path):
    #     path, tail = os.path.split(path)

    # if not os.path.isdir(path):
    #     if tail is not None:
    #         path = os.path.join(path, tail)
    #     raise IsADirectoryError("{} is not directory".format(path))

    head, _ = os.path.split(path)
    if not os.path.exists(head):
        os.makedirs(head)
    if not os.path.exists(path):
        os.makedirs(path)


def setup_file(path):
    head, _ = os.path.split(path)
    if not os.path.exists(head):
        os.makedirs(head)


def setup_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def teardown_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def open_chrome(url):
    webbrowser.open(url)


def print_lines(lines, max_line=50, split_print=True):
    remain_lines = len(lines)
    line_count = 0
    for line in lines:
        print("<{}>[{}]".format(line_count, line))
        line_count += 1
        remain_lines -= 1

        if split_print and line_count == max_line:
            line_count = 0
            input("remain {} lines, press key to continue".format(remain_lines))

    print("print lines end\n")


def path_join(*args):
    return os.path.join(*args)


def log_error_trace(log_func, e, head=""):
    log_func(f'{head}\n{error_trace(e)}')


def error_trace(e):
    exc_type, exc_value, exc_traceback = sys.exc_info()

    msg = '%s %s : %s\n\n' % (
        "".join(traceback.format_tb(exc_traceback)),
        e.__class__.__name__,
        e,
    )
    return msg


def params_to_dict(**param):
    return param


class temp_directory:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        setup_directory(self.path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if (exc_type, exc_val, exc_tb) == (None, None, None):
            teardown_directory(self.path)
            return True
        else:
            print(exc_tb)
            return False


def var_info(var, full=False, limit_lines=5):
    previous_frame = inspect.currentframe().f_back
    (filename, line_number, function_name, lines, index) = inspect.getframeinfo(previous_frame)

    s = "\n"

    var_str = str(var)
    split = var_str.split('\n')
    if not full and len(split) > limit_lines:
        var_str = "\n".join(split[:limit_lines])
        s += f'{var_str}\n'
        s += f'...\n'
        s += f'line exceed {limit_lines} \n'
    else:
        s += f'{var_str}\n'

    s += f'type = {type(var)}\n'

    if hasattr(var, '__len__'):
        s += f'len = {len(var)}\n'

    if hasattr(var, '__name__'):
        s += f'name = {var.__name__}\n'

    s += f'from {filename}\n'
    s += f'in line {line_number}, {lines}\n\n'

    return s


def lazy_property(fn):
    '''
    Decorator that makes a property lazy-evaluated.
    '''
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


def to_dict(**kwargs):
    return kwargs