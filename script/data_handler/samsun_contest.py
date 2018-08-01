import os
import pandas as pd
from script.data_handler.Base.BaseDataset import BaseDataset
from script.data_handler.Base.BaseDatasetPack import BaseDatasetPack
from script.data_handler.Base.Base_dfCleaner import Base_dfCleaner
from script.data_handler.Base.Base_df_transformer import Base_df_transformer
from script.data_handler.Base.base_df_typecasting import base_df_typecasting
from script.util.misc_util import path_join
from script.util.numpy_utils import *
from script.util.pandas_util import df_to_np_dict, df_to_onehot_embedding, df_to_np_onehot_embedding

DF = pd.DataFrame
Series = pd.Series




class train(BaseDataset):

    def load(self, path, limit=None):
        path = 'main_data'
        pd.read_csv()



        pass

    def transform(self):
        pass


class samsung_contest(BaseDatasetPack):
    pass
