from script.data_handler.wine_quality import wine_qualityPack


def test_get_set_item_BaseDataset():
    dataset_path = """C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\data\\winequality"""
    dataset_pack = wine_qualityPack().load(dataset_path)
    dataset_pack.shuffle()

    dataset_pack.split('data', 'train', 'test', rate=(7, 3))
    train_set = dataset_pack['train']
    Xs = train_set['Xs']

    # print(Xs)
    # pprint(Xs.shape)

    train_set['Xs'] = Xs


def test_BaseDataset_nest_batch():
    dataset_path = """C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\data\\winequality"""
    dataset_pack = wine_qualityPack().load(dataset_path)
    dataset_pack.shuffle()

    dataset_pack.split('data', 'train', 'test', rate=(7, 3))
    train_set = dataset_pack['train']

    Xs, Ys = train_set.next_batch(100)
    # pprint(Xs)

    Xs, Ys = train_set.full_batch()
    # pprint(Xs)
