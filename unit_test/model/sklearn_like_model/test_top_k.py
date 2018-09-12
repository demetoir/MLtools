from script.model.sklearn_like_model.Top_k_save import Top_k_save


def test_top_k():
    class dummy_model:
        def save(self, path):
            print(f'model save at {path}')

    top_k_save = Top_k_save('./test_instance/max_best/top_k/', save_model=False)
    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, -1, -1, -1]:
        top_k_save(i, dummy_model())

    top_k_save = Top_k_save('./test_instance/min_best/top_k', max_best=False, save_model=False)
    for i in [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -10, -10, -10, 1, 1, 1]:
        top_k_save(i, dummy_model())
