# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

from script.util.Logger import pprint_logger, Logger
from script.util.PlotTools import PlotTools
from script.util.deco import deco_timeit
# print(built-in function) is not good for logging
from script.workbench.TGS_salt_inference import cnn_pipeline, Unet_pipeline
from slackbot.SlackBot import deco_slackbot

bprint = print
logger = Logger('bench_code', level='INFO', )
print = logger.info
pprint = pprint_logger(print)
NpArr = np.array
DF = pd.DataFrame
Series = pd.Series
plot = PlotTools(save=True, show=False)


# plot = PlotTools(save=False, show=True)


def is_white_image(image):
    if np.mean(image) == 255:
        return True
    else:
        return False


def is_black_image(image):
    if np.mean(image) == 0:
        return True
    else:
        return False


class ExperimentResult:
    def __init__(self):
        self.elapse_time = None
        self.start_time = None
        self.finish_time = None
        self.id = None
        self.model_name = None
        self.epoch = None
        self.dataset = None
        self.git_version = None
        self.machine_info = None

    def __str__(self):
        pass

    def _param_dict(self, **kwargs):
        return kwargs


def get_platform_info():
    import platform

    info = {
        'architecture': platform.architecture(),
        'machine': platform.machine(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'system': platform.system(),
        'version': platform.version(),
        'uname': platform.uname(),
        'win32_ver': platform.win32_ver(),
        'java_ver': platform.java_ver(),
        'mac_ver': platform.mac_ver(),
    }
    return info


def deco_sigint_catch():
    # TODO
    import signal
    import sys

    def signal_handler(signal, frame):  # SIGINT handler정의
        print('You pressed Ctrl+C!')
        # DB정리하는 코드
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)  # 등록
    print('Press Ctrl+C')
    signal.pause()
    pass


@deco_timeit
@deco_slackbot('./slackbot/tokens/ml_bot_token', 'mltool_bot')
def main():
    # cnn_pipeline().train(100)
    Unet_pipeline().train(100, augmentation=False, early_stop=True, patience=20)
    pass
