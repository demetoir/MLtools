from script.data_handler.Base_df_null_handler import Base_df_null_handler


class HousePrices_null_handler(Base_df_null_handler):
    def BsmtCond(self, df):
        key = 'BsmtCond'
        col = df[[key]]
        series = df[key]
        self.print_null_col_info(df, key)

        df = self.fill_major_value(df, key)

        return df

    def BsmtExposure(self, df):
        key = 'BsmtExposure'
        col = df[[key]]
        series = df[key]
        self.print_null_col_info(df, key)

        # df = self.fill_major_value(df, key)

        return df

    def Fence(self, df):
        # may have corr.
        key = 'Fence'
        col = df[[key]]
        series = df[key]
        self.print_null_col_info(df, key)

        df = df.drop(columns='Fence')
        return df

    def Alley(self, df):
        # future corr test
        key = 'Alley'
        col = df[[key]]
        series = df[key]
        self.print_null_col_info(df, key)

        df = df.drop(columns='Alley')
        return df

    def MiscFeature(self, df):
        # future corr test
        key = 'MiscFeature'
        col = df[[key]]
        series = df[key]
        self.print_null_col_info(df, key)

        df = df.drop(columns='MiscFeature')
        return df

    def PoolQC(self, df):
        key = 'PoolQC'
        col = df[[key]]
        series = df[key]
        self.print_null_col_info(df, key)

        df = df.drop(columns='PoolQC')

        return df
