# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

from script.util.Logger import pprint_logger, Logger
from script.util.PlotTools import PlotTools
from script.util.deco import deco_timeit
from slackbot.SlackBot import deco_slackbot

bprint = print
logger = Logger('bench_code', level='INFO', )
print = logger.info
pprint = pprint_logger(print)
NpArr = np.array
DF = pd.DataFrame
Series = pd.Series
plot = PlotTools(save=True, show=False)


@deco_timeit
@deco_slackbot('./slackbot/tokens/ml_bot_token', 'mltool_bot')
def main():

    pass
