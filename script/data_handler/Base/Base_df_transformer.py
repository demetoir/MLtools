import pandas as pd
import inspect

from script.data_handler.Base.df_plotterMixIn import df_plotterMixIn
from script.util.MixIn import LoggerMixIn

DF = pd.DataFrame
Series = pd.Series


class Base_df_transformer(LoggerMixIn, df_plotterMixIn):
    def __init__(self, df: DF, df_Xs_keys, df_Ys_key, silent=False, verbose=0):
        LoggerMixIn.__init__(self, verbose)
        df_plotterMixIn.__init__(self)

        self.df = df
        self.df_Xs_keys = df_Xs_keys
        self.df_Ys_key = df_Ys_key
        self.silent = silent

    def __method_template(self, df: DF, key: str, col: DF, series: Series, Xs_key: list, Ys_key: list):
        return df

    def boilerplate_maker(self, path=None, encoding='UTF8'):
        base_class_name = self.__class__.__name__

        import_code = f"""
        import pandas as pd
        import numpy as np
        import random
        from script.data_handler.{base_class_name} import {base_class_name} 

        DF = pd.DataFrame
        Series = pd.Series

        """
        code = [import_code]

        base_class_name = self.__class__.__name__
        class_name = f"boiler_plate_{base_class_name}"
        class_template = f"""class {class_name}({base_class_name}):"""
        code += [class_template.format(class_name=class_name)]

        method_template = inspect.getsource(self.__method_template)
        method_template = method_template.replace('__method_template', '{col_name}')
        for key in self.df.keys():
            method_code = method_template.format(col_name=key)
            code += [method_code]

        code = "\n".join(code)
        if path is not None:
            with open(path, mode='w', encoding=encoding) as f:
                f.write(code)

        return code
