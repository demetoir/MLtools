import pandas as pd
import numpy as np
import random

DF = pd.DataFrame


class Base_df_null_handler:
    def __init__(self, df: DF, df_Xs_keys, df_Ys_key, silent=False):
        self.df = df
        self.silent = silent
        self.df_Xs_keys = df_Xs_keys
        self.df_Ys_keys = df_Ys_key

    def execute(self, *args, **kwargs) -> DF:
        for key, val in self.__class__.__dict__.items():
            if key in self.df.keys():
                self.df = val(self, self.df)

        return self.df

    @staticmethod
    def drop_col(df: DF, key):
        return df.drop(columns=key)

    @staticmethod
    def fill_major_value(df: DF, key):
        major_value = df[key].describe()['top']
        df[key] = df[key].fillna(major_value)
        return df

    @staticmethod
    def fill_random_value(df: DF, key):
        values = df[key].value_counts().keys()
        df[key] = df[key].fillna(lambda x: random.choice(values))
        return df

    @staticmethod
    def fill_rate_value(df: DF, key):
        values, count = zip(*list(df[key].value_counts().items()))
        p = np.array(count) / np.sum(count)
        df[key] = df[key].fillna(lambda x: random.choice(values, p=p))
        return df

    def boilerplate_maker(self, df, templete=None, path=None):
        class_name = "boilder_plate_Base_df_Null_handler"
        class_templete = """class {class_name}(Base_df_Null_handler):"""

        mothod_templete = """
            def {col_name}(self, df, df_Ys_key):
                key = '{col_name}'
                col = df[[key]]
                series = df[key]
                self.print_null_col_info(df, key, df_Ys_key)

                # idx = df[{col_name}].isna().to_list()

                # df = df.drop(columns='BsmtCond')
                return df
        """

        code = [class_templete.format(class_name=class_name)]
        for key in df.keys():
            method_code = mothod_templete.format(col_name=key)
            code += [method_code]

        code = "\n".join(code)
        if path is not None:
            with open(path, mode='w', encoding='UTF8') as f:
                f.write(code)

        return code

    def print_null_col_info(self, df: DF, key):
        if not self.silent:
            col = df[[key]]
            series = df[key]
            print()

            na_count = series.isna().sum()
            total = len(col)
            print(f'column : "{key}", {float(na_count)/float(total):.4f}% {na_count}/{total}(null/total)')
            print(col.describe())

            print('value_counts')
            print(series.value_counts())

            groupby = df[[key, self.df_Ys_keys]].groupby(key).agg(['mean', 'std', 'min', 'max', 'count'])
            print(groupby)

            print()

    def plot_info(self, df: DF, path=None):

        pass
