# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from script.util.Logger import pprint_logger, Logger
from script.util.deco import deco_timeit

# print(built-in function) is not good for logging
bprint = print
logger = Logger('bench_code', level='INFO', )
print = logger.info
pprint = pprint_logger(print)
NpArr = np.array
DF = pd.DataFrame


@deco_timeit
def main():
    pass
