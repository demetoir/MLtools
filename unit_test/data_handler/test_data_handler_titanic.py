from script.data_handler.titanic import trans_cabin, load_merge_set


def test_trans_cabin():
    merge_set_df = load_merge_set()
    cabin = trans_cabin(merge_set_df)

    head = cabin.head(5)
    # pprint(head)
    info = cabin.info()
    # pprint(info)
