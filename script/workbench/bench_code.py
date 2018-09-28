# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

from script.util.Logger import pprint_logger, Logger
from script.util.PlotTools import PlotTools
from script.util.deco import deco_timeit
# print(built-in function) is not good for logging
from script.workbench.TGS_salt.SS_Inference import SemanticSegmentation_pipeline
from script.workbench.TGS_salt.is_empty_mask_inference import is_emtpy_mask_clf_pipeline
from script.workbench.TGS_salt.mask_rate_inference import mask_rate_reg_pipeline
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


def SS():
    params = {
        'depth': [1, 2, 3, 4],
        'capacity': [8, 16, 24, 32, 64],
        'batch_size': [8, 16, 32, 64, 128],
        'loss_type': ['iou', 'dice_soft', 'pixel_wise_softmax', 'combine'],
        'net_type': ['FusionNet', 'UNet', 'InceptionUNet'],
    }

    pipe = SemanticSegmentation_pipeline()
    param = pipe.params(
        depth=2,
        batch_size=32,
        # net_type='InceptionUNet',
        net_type='FusionNet',
        capacity=64,
        learning_rate=0.01,
        loss_type='BCE+dice_soft',
        dropout_rate=0.5,
        comment='depth =2 and atrous conv and dropout, double capacity',
    )
    pipe.train(param, n_epoch=100, augmentation=False, early_stop=True, patience=20, )


def is_empty_mask():
    pipe = is_emtpy_mask_clf_pipeline()
    params = pipe.params(
        capacity=4,
        # net_type='InceptionV2',
        net_type='ResNet18',
        # batch_size=32,
        batch_size=256,
        learning_rate=0.002,
        dropout_rate=0.5,
        fc_capacity=1024,
        fc_depth=2,
        comment='testing',
    )
    pipe.train(params, n_epoch=30, augmentation=False, early_stop=True)


def mask_reg():
    pipe = mask_rate_reg_pipeline()
    params = pipe.params(
        capacity=8,
        net_type='InceptionV1',
        loss_type='MSE',
        learning_rate=0.01,
        batch_size=32,
        comment='non_empty_mask_only_train'
    )
    pipe.train(params, n_epoch=100)


@deco_timeit
@deco_slackbot('./slackbot/tokens/ml_bot_token', 'mltool_bot')
def main():
    SS()
    # mask_reg()
    # is_empty_mask()
