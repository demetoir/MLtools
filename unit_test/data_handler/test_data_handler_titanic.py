from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.data_handler.titanic import trans_cabin, load_merge_set


def test_trans_cabin():
    merge_set_df = load_merge_set()
    cabin = trans_cabin(merge_set_df)

    head = cabin.head(5)
    # pprint(head)
    info = cabin.info()
    # pprint(info)


def test_titanic_pack():
    datapack = DatasetPackLoader().load_dataset("titanic")
    train_set = datapack['train']

    train_set, valid_set = train_set.split((7, 3))
    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])
    sample_Xs, sample_Ys = valid_Xs[:2], valid_Ys[:2]
    print(train_Xs)

