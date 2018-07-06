from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.model.sklearn_like_model.GAN.InfoGAN import InfoGAN
from pprint import pprint


def to_zero_one_encoding(x):
    x[x >= 0.5] = 1.0
    x[x < 0.5] = 0.0
    return x


def load_dataset():
    data_pack = DatasetPackLoader().load_dataset('titanic')
    train_set = data_pack['train']
    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    return train_Xs, train_Ys


def common_info_GAN(gan_cls, params, epoch=1):
    train_Xs, train_Ys = load_dataset()

    gan = gan_cls(**params)

    gan.train(train_Xs, train_Ys, epoch=epoch)

    metric = gan.metric(train_Xs, train_Ys)
    pprint(metric)

    gen = gan.generate(train_Ys[:2])

    gen = to_zero_one_encoding(gen)
    pprint(gen)

    path = gan.save()

    gan = gan_cls()
    gan.load(path)
    gan.train(train_Xs, train_Ys, epoch=epoch)

    metric = gan.metric(train_Xs, train_Ys)
    pprint(metric)

    gen = gan.generate(train_Ys[:2])
    gen = to_zero_one_encoding(gen)
    pprint(gen)


def test_Info_GAN_loss_GAN():
    gan = InfoGAN
    params = {
        'loss_type': 'GAN'
    }
    epoch = 1
    common_info_GAN(gan, params, epoch=epoch)


def test_Info_GAN_loss_WGAN():
    gan = InfoGAN
    params = {
        'loss_type': 'WGAN'
    }
    epoch = 1
    common_info_GAN(gan, params, epoch=epoch)


def test_Info_GAN_loss_LSGAN():
    gan = InfoGAN
    params = {
        'loss_type': 'LSGAN'
    }
    epoch = 1
    common_info_GAN(gan, params, epoch=epoch)


def test_Info_GAN_loss_L1_GAN():
    gan = InfoGAN
    params = {
        'loss_type': 'L1_GAN'
    }
    epoch = 1
    common_info_GAN(gan, params, epoch=epoch)
