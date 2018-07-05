from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.model.sklearn_like_model.GAN.C_GAN import C_GAN


def to_zero_one_encoding(x):
    x[x >= 0.5] = 1.0
    x[x < 0.5] = 0.0
    return x


def load_dataset():
    datapack = DatasetPackLoader().load_dataset('titanic')
    train_set = datapack['train']
    test_set = datapack['test']
    train_set.shuffle()
    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    sample_Xs, sample_Ys = train_Xs[:1], train_Ys[:2]
    return train_Xs, train_Ys


def GAN_common_titanic(gan_cls, params):
    train_Xs, train_Ys = load_dataset()

    gan = gan_cls(**params)

    gan.train(train_Xs, train_Ys, epoch=1)

    metric = gan.metric(train_Xs, train_Ys)
    # pprint(metric)

    gen = gan.generate(2, [[1, 0], [1, 0]])

    gen = to_zero_one_encoding(gen)
    # pprint(gen)

    path = gan.save()

    gan = gan_cls()
    gan.load(path)
    gan.train(train_Xs, train_Ys, epoch=1)

    metric = gan.metric(train_Xs, train_Ys)
    # pprint(metric)

    gen = gan.generate(2, [[1, 0], [1, 0]])
    gen = to_zero_one_encoding(gen)
    # pprint(gen)


def test_C_GAN_GAN_loss():
    params = {
        'loss_type': 'GAN'
    }
    GAN_common_titanic(C_GAN, params)


def test_C_GAN_WGAN_loss():
    params = {
        'loss_type': 'WGAN'
    }
    GAN_common_titanic(C_GAN, params)


def test_C_GAN_LSGAN_loss():
    params = {
        'loss_type': 'LSGAN'
    }
    GAN_common_titanic(C_GAN, params)


def test_C_GAN_L1GAN_loss():
    params = {
        'loss_type': 'L1_GAN'
    }
    GAN_common_titanic(C_GAN, params)
