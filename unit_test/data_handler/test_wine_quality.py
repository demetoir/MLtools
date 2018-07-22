from pprint import pprint
from script.data_handler.wine_quality import load_merge_data, wine_quality_transformer, wine_qualityPack


def test_wine_quality_transformer():
    df_Xs_keys = [
        'col_0_fixed_acidity', 'col_1_volatile_acidity', 'col_2_citric_acid',
        'col_3_residual_sugar', 'col_4_chlorides', 'col_5_free_sulfur_dioxide',
        'col_6_total_sulfur_dioxide', 'col_7_density', 'col_8_pH',
        'col_9_sulphates', 'col_10_alcohol', 'col_12_color'
    ]
    df_Ys_key = 'col_11_quality'

    merge_df = load_merge_data(cache=False)

    print(merge_df.keys())
    transformer = wine_quality_transformer(merge_df, df_Xs_keys, df_Ys_key)
    transformer.boilerplate_maker('./gen_code.py')
    merge_df = transformer.transform()
    print(merge_df.info())
    transformer.plot_all(merge_df, merge_df.keys(), df_Ys_key)


def load_wine_quality_dataset():
    dataset_path = """C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\data\\winequality"""
    dataset_pack = wine_qualityPack().load(dataset_path)
    dataset_pack.shuffle()
    dataset_pack.split('data', 'train', 'test', rate=(7, 3))
    train_set = dataset_pack['train']
    test_set = dataset_pack['test']

    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    test_Xs, test_Ys = test_set.full_batch(['Xs', 'Ys'])
    sample_xs = train_Xs[:5]
    sample_Ys = train_Ys[:5]
    return train_Xs, train_Ys, test_Xs, test_Ys, sample_xs, sample_Ys


def test_wine_quality_dataset_pack():
    dataset_path = """C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\data\\winequality"""
    dataset_pack = wine_qualityPack().load(dataset_path)

    dataset = dataset_pack['data']
    df = dataset.to_DataFrame()
    print(df.info())

    dataset.shuffle(random_state=7)
    train_set, test_set = dataset.split((7, 3))

    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    test_Xs, test_Ys = train_set.full_batch(['Xs', 'Ys'])

    pprint(train_Xs.shape)
    pprint(train_Ys.shape)
    pprint(test_Xs.shape)
    pprint(test_Ys.shape)
