from data_handler.DatasetPackLoader import DatasetPackLoader
from model.sklearn_like_model.GAN.C_GAN import C_GAN


def to_zero_one_encoding(x):
    x[x >= 0.5] = 1.0
    x[x < 0.5] = 0.0
    return x


def test_C_GAN():
    datapack = DatasetPackLoader().load_dataset('titanic')
    train_set = datapack['train']
    test_set = datapack['test']
    train_set.shuffle()

    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    sample_Xs, sample_Ys = train_Xs[:1], train_Ys[:2]

    gan = C_GAN()
    gan.train(train_Xs, train_Ys, epoch=1)

    metric = gan.metric(train_Xs, train_Ys)
    # pprint(metric)

    gen = gan.generate(2, [[1, 0], [1, 0]])

    gen = to_zero_one_encoding(gen)
    # pprint(gen)

    path = gan.save()
    gan = C_GAN()
    gan.load(path)
    gan.train(train_Xs, train_Ys, epoch=1)

    metric = gan.metric(train_Xs, train_Ys)
    # pprint(metric)

    gen = gan.generate(2, [[1, 0], [1, 0]])
    gen = to_zero_one_encoding(gen)
    # pprint(gen)
