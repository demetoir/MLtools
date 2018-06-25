"""env setting"""
import os

FILE = 'env_setting.py'
ENV_SETTING_PATH = os.path.dirname(os.path.realpath(FILE))
ROOT_PATH = os.path.join(ENV_SETTING_PATH)

# model
MODEL_MODULE_PATH = os.path.join(ROOT_PATH, 'script', 'model')

# visualizer
VISUALIZER_MODULE_PATH = os.path.join(ROOT_PATH, 'script', 'visualizer')

# dataset handler
DATA_HANDLER_PATH = os.path.join(ROOT_PATH, 'script', 'data_handler')

# dataset
DATA_PATH = os.path.join(ROOT_PATH, 'data')

# instance
INSTANCE_PATH = os.path.join(ROOT_PATH, 'instance')

# log
LOG_PATH = os.path.join(ROOT_PATH, 'log')

# sklearn_params
SKLEARN_PARAMS_SAVE_PATH = os.path.join(ROOT_PATH, 'sklearn_param_save')


def tensorboard_dir():
    import tensorboard
    path, _ = os.path.split(tensorboard.__file__)
    tensorboard_main = os.path.join(path, 'main.py')
    del tensorboard
    return tensorboard_main
