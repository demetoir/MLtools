from script.data_handler.DFEncoder import DFEncoder
from pprint import pprint
from pandas import DataFrame as DF


def test_DFEncoder():
    df = DF({
        'A': ['a', 'b', 'c', 'a', 'c'],
        'B': [1, 2, 3, 4, 5]
    })

    enc = DFEncoder()
    cate_cols = ['A']
    conti_cols = ['B']
    enc.fit(df)

    encoder_df = enc.encode(df)
    pprint(list(encoder_df.columns))
    pprint(encoder_df.head())

    decoded_df = enc.decode(encoder_df)
    pprint(decoded_df.info())
    pprint(decoded_df.head())
    print(df[conti_cols].head())

    np_arr = enc.to_np(encoder_df)
    print(np_arr)

    encoder_df = enc.from_np(np_arr)
    pprint(encoder_df.info())
    pprint(encoder_df.head())

    enc.dump('./enc.pckl')
    enc = DFEncoder().load('./enc.pckl')

    cate_cols = ['A']
    conti_cols = ['B']
    encoder_df = enc.encode(df)
    pprint(list(encoder_df.columns))
    pprint(encoder_df.head())

    decoded_df = enc.decode(encoder_df)
    pprint(decoded_df.info())
    pprint(decoded_df.head())
    print(df[conti_cols].head())

    np_arr = enc.to_np(encoder_df)
    print(np_arr)

    encoder_df = enc.from_np(np_arr)
    pprint(encoder_df.info())
    pprint(encoder_df.head())
