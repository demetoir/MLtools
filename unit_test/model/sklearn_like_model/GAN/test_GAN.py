from script.model.sklearn_like_model.GAN.GAN import GAN
from script.data_handler.DatasetPackLoader import DatasetPackLoader


def to_zero_one_encoding(x):
    x[x >= 0.5] = 1.0
    x[x < 0.5] = 0.0
    return x


def load_dataset():
    datapack = DatasetPackLoader().load_dataset('titanic')
    train_set = datapack['train']
    train_set.shuffle()
    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    return train_Xs


def GAN_common_titanic(gan_cls, params):
    train_Xs = load_dataset()

    gan = gan_cls(**params)

    gan.train(train_Xs, epoch=1)

    metric = gan.metric(train_Xs)
    # pprint(metric)

    gen = gan.generate(2)

    gen = to_zero_one_encoding(gen)
    # pprint(gen)

    path = gan.save()

    gan = gan_cls()
    gan.load(path)
    gan.train(train_Xs, epoch=1)

    metric = gan.metric(train_Xs)
    # pprint(metric)

    gen = gan.generate(2)
    gen = to_zero_one_encoding(gen)
    # pprint(gen)


def test_GAN_GAN_loss():
    params = {
        'loss_type': 'GAN'
    }
    GAN_common_titanic(GAN, params)


def test_GAN_WGAN_loss():
    params = {
        'loss_type': 'WGAN'
    }
    GAN_common_titanic(GAN, params)


def test_GAN_LSGAN_loss():
    params = {
        'loss_type': 'LSGAN'
    }
    GAN_common_titanic(GAN, params)
